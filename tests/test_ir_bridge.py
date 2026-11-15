"""The mlopt-IR -> kernel-IR bridge: honest subset, loud declines.

``lib_kernel_ir`` is a git dep (tag v0.1.0); dev worktrees may resolve
a sibling ``../lib-kernel-ir`` checkout ahead of the installed wheel,
so a clone without it skips this file honestly rather than erroring at
collection (same pattern as ``test_kernel_search_ir.py``).
"""

from __future__ import annotations

import pytest
import torch

lib_kernel_ir = pytest.importorskip("lib_kernel_ir")

from lib_kernel_ir import Leaf, Term, from_json, to_json  # noqa: E402
from mlopt_core.ir import Const, Op, Param, TensorType, Var  # noqa: E402

from mlopt_kernel.ir_bridge import UnsupportedOp, translate  # noqa: E402

K, N = 8, 16
V = Var("v", TensorType((K,)))
W = Var("w", TensorType((K, N)))
B = Param("b", TensorType((N,)))
X = Var("x", TensorType((4, 8)))
Y = Var("y", TensorType((4, 8)))
# ADR-0171 spellings: weight-first GEMV, F.linear, layout, pointwise.
WM = Var("wm", TensorType((4, K)))  # [m, k]
WL = Var("wl", TensorType((N, K)))  # [out, in]
Z = Var("z", TensorType((2, 3, 4)))
U = Var("u", TensorType(None))  # untyped — rank unknown


def env() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "v": torch.randn(K, generator=g),
        "w": torch.randn(K, N, generator=g),
        "b": torch.randn(N, generator=g),
        "x": torch.randn(4, 8, generator=g),
        "y": torch.randn(4, 8, generator=g),
        "wm": torch.randn(4, K, generator=g),
        "wl": torch.randn(N, K, generator=g),
        "z": torch.randn(2, 3, 4, generator=g),
    }


def test_gemv_add_roundtrip_denotes_and_costs() -> None:
    """add(matmul(v[k], M[k,n]), b[n]) -> bias(dot(M, v), b)."""
    mlopt_term = Op.make("add", Op.make("matmul", V, W), B)
    kit = translate(mlopt_term)

    assert isinstance(kit, Term)
    assert kit.op == "bias"
    dot, bias_leaf = kit.args
    assert isinstance(dot, Term) and dot.op == "dot"
    # dot(matrix, vector): the matrix arg leads in kernel-ir.
    assert dot.args == (Leaf("w"), Leaf("v"))
    assert bias_leaf == Leaf("b")

    e = env()
    expected = torch.matmul(e["v"], e["w"]) + e["b"]
    out = kit.denote(e)
    assert out.shape == (N,)
    torch.testing.assert_close(out, expected)

    cost = kit.cost(e)
    assert cost.bytes_moved > 0 and cost.launches >= 0


def test_k_leaves_mark_split_slices() -> None:
    """``k_leaves`` gives the dot operands ``kind="k"`` for split_k."""
    mlopt_term = Op.make("matmul", V, W)
    kit = translate(mlopt_term, k_leaves={"v", "w"})
    assert isinstance(kit, Term)
    matrix, vec = kit.args
    assert matrix == Leaf("w", "k")
    assert vec == Leaf("v", "k")

    # A hand-wrapped split_k: the k-partitioned dot must still denote
    # the full GEMV — the property split_k exists to preserve.
    split_term = Term("split", (K, 2, kit))
    e = env()
    torch.testing.assert_close(
        split_term.denote(e), torch.matmul(e["v"], e["w"])
    )


def test_default_leaves_are_unsliced() -> None:
    kit = translate(Op.make("matmul", V, W))
    assert kit.args == (Leaf("w"), Leaf("v"))
    assert all(arg.kind == "out" for arg in kit.args)


def test_json_roundtrip() -> None:
    kit = translate(Op.make("add", Op.make("matmul", V, W), B))
    assert from_json(to_json(kit)) == kit


def test_bare_var_is_a_leaf() -> None:
    assert translate(V) == Leaf("v")
    assert translate(B) == Leaf("b")


def test_matmul_weight_first_maps_through_transpose() -> None:
    """[m,k] @ [k] -> [m] is dot(transpose(W), v) — ADR-0171."""
    kit = translate(Op.make("matmul", WM, V))

    assert isinstance(kit, Term)
    assert kit.op == "dot"
    weight, vec = kit.args
    assert isinstance(weight, Term) and weight.op == "transpose"
    assert weight.args == (Leaf("wm"), 0, 1)
    assert vec == Leaf("v")

    e = env()
    torch.testing.assert_close(
        kit.denote(e), torch.matmul(e["wm"], e["v"])
    )
    assert kit.cost(e).bytes_moved > 0
    assert from_json(to_json(kit)) == kit


def test_matmul_weight_first_dim_mismatch_declines() -> None:
    bad = Var("bad", TensorType((4, K + 1)))
    with pytest.raises(UnsupportedOp, match="contracted dims"):
        translate(Op.make("matmul", bad, V))


def test_matmul_general_declines() -> None:
    """[4,8] @ [8,4] -> [4,4] is not a dot."""
    with pytest.raises(UnsupportedOp, match="GEMV"):
        translate(Op.make("matmul", X, Var("w2", TensorType((8, 4)))))


def test_matmul_contracted_dim_mismatch_declines() -> None:
    bad = Var("bad", TensorType((K + 1, N)))
    with pytest.raises(UnsupportedOp, match="contracted dims"):
        translate(Op.make("matmul", V, bad))


def test_linear_maps_with_and_without_bias() -> None:
    """linear(v[in], W[out,in], b) -> bias(dot(Wᵀ, v), b)."""
    e = env()
    with_bias = translate(Op.make("linear", V, WL, B))
    assert isinstance(with_bias, Term) and with_bias.op == "bias"
    dot, bias_leaf = with_bias.args
    assert isinstance(dot, Term) and dot.op == "dot"
    weight, vec = dot.args
    assert isinstance(weight, Term) and weight.op == "transpose"
    assert weight.args == (Leaf("wl"), 0, 1)
    assert vec == Leaf("v") and bias_leaf == Leaf("b")
    torch.testing.assert_close(
        with_bias.denote(e),
        torch.nn.functional.linear(e["v"], e["wl"], e["b"]),
    )

    no_bias = translate(Op.make("linear", V, WL))
    assert isinstance(no_bias, Term) and no_bias.op == "dot"
    torch.testing.assert_close(
        no_bias.denote(e),
        torch.nn.functional.linear(e["v"], e["wl"]),
    )


def test_linear_batched_input_declines() -> None:
    """[batch, in] @ [out, in] is a matmul, not a dot."""
    with pytest.raises(UnsupportedOp, match="rank-1"):
        translate(Op.make("linear", X, WL))


def test_transpose_maps_attrs_to_positional_dims() -> None:
    kit = translate(Op.make("transpose", W, dim0=0, dim1=1))
    assert kit == Term("transpose", (Leaf("w"), 0, 1))
    e = env()
    torch.testing.assert_close(kit.denote(e), e["w"].t())

    # No attrs -> the torch binding's (-2, -1) default.
    kit_default = translate(Op.make("transpose", W))
    assert kit_default == Term("transpose", (Leaf("w"), -2, -1))
    torch.testing.assert_close(kit_default.denote(e), e["w"].t())


def test_transpose_rejects_non_int_axis() -> None:
    term = Op.make("transpose", W, dim0="x", validate=False)
    with pytest.raises(UnsupportedOp, match="not an int axis"):
        translate(term)


def test_permute_maps_dims_attr() -> None:
    kit = translate(Op.make("permute", Z, dims=[2, 0, 1]))
    assert kit == Term("permute", (Leaf("z"), 2, 0, 1))
    e = env()
    torch.testing.assert_close(kit.denote(e), e["z"].permute(2, 0, 1))


def test_permute_no_dims_reverses_all_axes() -> None:
    """mlopt's permute() convention: reverse — needs a known rank."""
    kit = translate(Op.make("permute", Z))
    assert kit == Term("permute", (Leaf("z"), 2, 1, 0))
    e = env()
    torch.testing.assert_close(kit.denote(e), e["z"].permute(2, 1, 0))


def test_permute_no_dims_unknown_rank_declines() -> None:
    with pytest.raises(UnsupportedOp, match="unknown rank"):
        translate(Op.make("permute", U, validate=False))


def test_permute_wrong_dim_count_declines() -> None:
    term = Op.make("permute", Z, dims=[0, 1], validate=False)
    with pytest.raises(UnsupportedOp, match="rank-3"):
        translate(term)


@pytest.mark.parametrize(
    "op_name,args",
    [
        ("mul", (X, Y)),
        ("sub", (X, Y)),
        ("neg", (X,)),
        ("relu", (X,)),
        ("exp", (X,)),
        ("sigmoid", (X,)),
        ("tanh", (X,)),
        ("gelu", (X,)),
        ("silu", (X,)),
    ],
)
def test_pointwise_ops_map_name_for_name(op_name, args) -> None:
    term = Op.make(op_name, *args, validate=False)
    kit = translate(term)
    assert isinstance(kit, Term) and kit.op == op_name
    leaves = [a.name for a in kit.args if isinstance(a, Leaf)]
    assert leaves == [a.name for a in args]

    e = env()
    want = getattr(torch, op_name, None)
    if want is None:
        want = getattr(torch.nn.functional, op_name)
    if len(args) == 2:
        want = want(e["x"], e["y"])
    else:
        want = want(e["x"])
    torch.testing.assert_close(kit.denote(e), want)
    assert kit.cost(e).ops > 0


def test_pointwise_with_const_declines() -> None:
    with pytest.raises(UnsupportedOp, match="scalar"):
        translate(Op.make("mul", X, Const(2.0), validate=False))


def test_pointwise_with_attrs_declines() -> None:
    term = Op.make("relu", X, approximate=True, validate=False)
    with pytest.raises(UnsupportedOp, match="attrs"):
        translate(term)


def test_gated_tail_denotes_end_to_end() -> None:
    """silu(mul(linear(v, W), u)) — a decoder-style tail, all bridged."""
    u = Var("u", TensorType((N,)))
    e = env()
    e["u"] = torch.randn(N, generator=torch.Generator().manual_seed(3))
    term = Op.make(
        "silu",
        Op.make("mul", Op.make("linear", V, WL), u, validate=False),
        validate=False,
    )
    kit = translate(term)
    want = torch.nn.functional.silu(
        torch.nn.functional.linear(e["v"], e["wl"]) * e["u"]
    )
    torch.testing.assert_close(kit.denote(e), want)


@pytest.mark.parametrize(
    "op_name",
    [
        "sum",
        "mean",
        "softmax",
        "conv2d",
        "split",  # name collision: mlopt split != kernel-ir split
        "reshape",
        "flatten",
        "div",
        "sqrt",
        "layer_norm",
        "sdpa",
        "rms_norm",
    ],
)
def test_no_counterpart_ops_decline_loudly(op_name: str) -> None:
    term = Op.make(op_name, X, Y, validate=False)
    with pytest.raises(UnsupportedOp) as exc:
        translate(term)
    assert exc.value.op == op_name
    assert "no lib-kernel-ir counterpart" in exc.value.reason


def test_add_with_const_declines() -> None:
    """Scalar adds denote fine but crash ``Term.cost`` — declined."""
    with pytest.raises(UnsupportedOp, match="Const"):
        translate(Op.make("add", X, Const(0.5)))
    with pytest.raises(UnsupportedOp, match="Const"):
        translate(Op.make("add", Const(1), Y))


def test_matmul_with_attrs_declines() -> None:
    term = Op.make("matmul", V, W, layout="foo", validate=False)
    with pytest.raises(UnsupportedOp, match="attrs"):
        translate(term)


def test_const_root_and_non_terms_decline() -> None:
    with pytest.raises(UnsupportedOp, match="scalar"):
        translate(Const(3))
    with pytest.raises(UnsupportedOp, match="not an mlopt_core"):
        translate(torch.randn(2))


def test_nested_unmappable_op_declines_inside() -> None:
    """add(sum(x), w): the inner op's decline propagates, named."""
    term = Op.make(
        "add",
        Op.make("sum", X, validate=False),
        Var("w1", TensorType((4, 8))),
    )
    with pytest.raises(UnsupportedOp) as exc:
        translate(term)
    assert exc.value.op == "sum"
