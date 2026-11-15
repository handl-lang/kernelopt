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


def env() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "v": torch.randn(K, generator=g),
        "w": torch.randn(K, N, generator=g),
        "b": torch.randn(N, generator=g),
        "x": torch.randn(4, 8, generator=g),
        "y": torch.randn(4, 8, generator=g),
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


def test_matmul_weight_first_declines_with_reason() -> None:
    """[m,k] @ [k] -> [m] needs a transpose the IR does not have."""
    wm = Var("wm", TensorType((4, K)))
    with pytest.raises(UnsupportedOp, match="transpose"):
        translate(Op.make("matmul", wm, V))


def test_matmul_general_declines() -> None:
    """[4,8] @ [8,4] -> [4,4] is not a dot."""
    with pytest.raises(UnsupportedOp, match="GEMV"):
        translate(Op.make("matmul", X, Var("w2", TensorType((8, 4)))))


def test_matmul_contracted_dim_mismatch_declines() -> None:
    bad = Var("bad", TensorType((K + 1, N)))
    with pytest.raises(UnsupportedOp, match="contracted dims"):
        translate(Op.make("matmul", V, bad))


@pytest.mark.parametrize(
    "op_name",
    [
        "mul",
        "sub",
        "relu",
        "sigmoid",
        "exp",
        "sum",
        "mean",
        "softmax",
        "linear",
        "conv2d",
        "split",  # name collision: mlopt split != kernel-ir split
        "reshape",
        "permute",
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
    """add(relu(x), w): the inner op's decline propagates, named."""
    term = Op.make(
        "add",
        Op.make("relu", X, validate=False),
        Var("w1", TensorType((4, 8))),
    )
    with pytest.raises(UnsupportedOp) as exc:
        translate(term)
    assert exc.value.op == "relu"
