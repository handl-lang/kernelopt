"""Bridge: ``mlopt_core.term`` terms -> ``lib_kernel_ir`` Terms.

Phase-C spike of the handl-bridge proposal: can a term the mlopt
pipeline extracts (``Optimizer`` phase 3, an ``mlopt_core.term.Op`` DAG)
be handed to kernelopt's codegen search as a ``lib_kernel_ir.Term``?

The honest answer is *a recognizer, not a translator* — and after
ADR-0171 (proposed; ``lib-kernel-ir`` gained ``transpose``/``permute``
plus the pointwise set ``mul``/``sub``/``neg``/``relu``/``exp``/
``sigmoid``/``tanh``/``gelu``/``silu``) the covered subset is:

* ``matmul`` — the GEMV spellings ``matmul(v[k], M[k, n]) -> [n]`` and
  ``matmul(W[m, k], v[k]) -> [m]`` both denote a ``lib_kernel_ir``
  ``dot`` (``einsum("kn,k->n")`` — contracted dim *first*; the
  weight-first spelling goes through ``transpose``).  Rank-2 @ rank-2
  and batched forms still decline: they are not a ``dot``.
* ``linear`` — ``F.linear(x[in], W[out, in], b?)`` with a rank-1 input
  is ``dot(transpose(W), x)`` plus an optional ``bias``.  Batched
  (rank > 1) inputs decline — a matrix-matrix product is not a dot.
* ``add`` — ``lib_kernel_ir``'s elementwise-add spelling is ``bias``
  (``tile + bias``, torch broadcasting, grade ROUNDS).  Only between
  two tensor operands: a ``Const`` operand would denote fine yet
  crash ``Term.cost`` (``_n`` needs ``numel``), so it declines.
* ``transpose``/``permute`` — layout views; dims come from the
  mlopt attrs (``dim0``/``dim1``, ``dim``/``dims``/``order``) and
  become positional int args on the kernel-ir op.
* ``mul``/``sub``/``neg``/``relu``/``exp``/``sigmoid``/``tanh``/
  ``gelu``/``silu`` — the pointwise tail, mapped name-for-name.

Everything else — ``sum``, ``mean``, ``softmax``, ``conv2d``,
``split`` (a *name* collision: mlopt's ``split`` cuts a tensor,
kernel-ir's partitions the k axis — declined, never confused),
``reshape``, ``flatten``, ``rms_norm``, ``sdpa``, casts, … — raises
:class:`UnsupportedOp` naming the op and the reason.  Nothing is
silently approximated.

    >>> from mlopt_kernel.ir_bridge import translate
    >>> term = Op.make("add", Op.make("matmul", v, w), b)
    >>> kit = translate(term, k_leaves={"v", "w"})
    >>> kit.denote({"v": x, "w": W, "b": bias})

Leaves: ``Var``/``Param`` become ``Leaf(name)``; ``Const`` is a
literal where an op accepts one (none of the mapped ops do, so a
``Const`` anywhere reachable declines).  ``k_leaves`` names the
variables a ``split_k`` search should slice on the k axis (``kind="k"``);
default is ``"out"`` — unsliced — which is correct but schedule-blind.

Known gaps (see ``docs/reports/mlopt_ir_kernel_bridge.md``): no dtype
or layout tracking (kernel-ir ``dot``/``dequant`` cast to fp32
env-side; mlopt preserves dtypes), the ``store``/``out``-buffer
epilogue is the caller's job, and reductions/norms/attention still
have no kernel-ir counterpart.
"""

from __future__ import annotations

from collections.abc import Container
from typing import Any

from lib_kernel_ir import Leaf, Term
from mlopt_core.term import Const, Op, Param, TensorType, Var

__all__ = ["UnsupportedOp", "translate"]


class UnsupportedOp(Exception):
    """An mlopt op (or term shape) with no honest kernel-ir spelling.

    ``op`` is the mlopt op name (or ``"<root>"`` for a non-Op term);
    ``reason`` says why the bridge declines — a missing counterpart,
    a layout the IR cannot express, or a cost-model hazard.
    """

    def __init__(self, op: str, reason: str) -> None:
        """Record the declined op name and the reason."""
        self.op = op
        self.reason = reason
        super().__init__(f"{op}: {reason}")


def _rank(typ: TensorType | None) -> int | None:
    """Rank of a leaf's tensor type; ``None`` when the type is unknown."""
    if typ is None or typ.shape is None:
        return None
    return len(typ.shape)


def _dim(typ: TensorType | None, i: int) -> int | None:
    """Dim ``i`` of a leaf's tensor type, or ``None`` if unknown."""
    if typ is None or typ.shape is None or i >= len(typ.shape):
        return None
    return typ.shape[i]


def _leaf_typ(node: Any) -> TensorType | None:
    """Return the declared tensor type of a Var/Param leaf, else ``None``."""
    if isinstance(node, (Var, Param)):
        return node.typ
    return None


def _shape(typ: TensorType | None) -> tuple:
    """Return the declared shape tuple, or ``()`` when unknown."""
    if typ is None or typ.shape is None:
        return ()
    return tuple(typ.shape)


def _leaf(node: Any, k_leaves: Container[str]) -> Leaf:
    """Return a Var/Param as a kernel-ir ``Leaf`` (kind ``"k"`` if requested)."""
    name = node.name
    return Leaf(name, "k" if name in k_leaves else "out")


def _tr_arg(
    node: Any, k_leaves: Container[str]
) -> Term | Leaf | int | float:
    """Translate one argument position of a mapped op."""
    if isinstance(node, (Var, Param)):
        return _leaf(node, k_leaves)
    if isinstance(node, Const):
        raise UnsupportedOp(
            "const",
            "kernel-ir cost model needs tensor operands "
            f"(got scalar {node.value!r})",
        )
    return translate(node, k_leaves=k_leaves)


def _check_no_attrs(term: Op) -> None:
    """Decline a mapped op carrying attrs the kernel-ir op cannot take."""
    if term.attrs:
        raise UnsupportedOp(
            term.op,
            f"attrs {sorted(term.attrs)} have no kernel-ir "
            "counterpart (the IR's ops take positional args only)",
        )


def _int_attr(
    term: Op, canonical: str, *alts: str, default: int | None = None
) -> int | None:
    """Read an int axis attr — canonical name first, ``argN`` spellings after."""
    for name in (canonical, *alts):
        if name not in term.attrs:
            continue
        value = term.attrs[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise UnsupportedOp(
                term.op, f"attr {name}={value!r} is not an int axis"
            )
        return value
    return default


def _matmul(term: Op, k_leaves: Container[str]) -> Term:
    """Map the GEMV spellings of ``matmul`` to ``dot`` — and only those.

    ``lib_kernel_ir.dot`` denotes ``einsum("kn,k->n")``: the contracted
    axis is dim 0 of *both* operands, output is dim 1 of the matrix.

    * ``matmul(v[k], M[k, n]) -> [n]`` — the matrix arg leads in
      kernel-ir: ``dot(M, v)``.
    * ``matmul(W[m, k], v[k]) -> [m]`` — torch contracts W's dim 1, so
      the weight goes through ``transpose`` first:
      ``dot(transpose(W, 0, 1), v)``.

    Rank-2 @ rank-2 and batched forms decline: they are not a ``dot``.
    """
    _check_no_attrs(term)
    if len(term.args) != 2:
        raise UnsupportedOp(
            "matmul", f"expected 2 args, got {len(term.args)}"
        )
    a, b = term.args
    ta, tb = _leaf_typ(a), _leaf_typ(b)
    ra, rb = _rank(ta), _rank(tb)
    # vector @ matrix: [k] @ [k, n] -> [n]
    if ra == 1 and rb == 2:
        ka, kb = _dim(ta, 0), _dim(tb, 0)
        if ka is not None and kb is not None and ka != kb:
            raise UnsupportedOp(
                "matmul",
                f"contracted dims differ: v{_shape(ta)} vs M{_shape(tb)}",
            )
        # dot(matrix, vector): the matrix arg comes first in kernel-ir.
        return Term("dot", (_tr_arg(b, k_leaves), _tr_arg(a, k_leaves)))
    # matrix @ vector: [m, k] @ [k] -> [m] — the common weight-first GEMV.
    if ra == 2 and rb == 1:
        mk, kv = _dim(ta, 1), _dim(tb, 0)
        if mk is not None and kv is not None and mk != kv:
            raise UnsupportedOp(
                "matmul",
                f"contracted dims differ: W{_shape(ta)} "
                f"vs v{_shape(tb)}",
            )
        weight = Term("transpose", (_tr_arg(a, k_leaves), 0, 1))
        return Term("dot", (weight, _tr_arg(b, k_leaves)))
    raise UnsupportedOp(
        "matmul",
        f"only the [k] @ [k, n] -> [n] and [m, k] @ [k] -> [m] GEMV "
        f"spellings map to dot; got ranks ({ra}, {rb})",
    )


def _add(term: Op, k_leaves: Container[str]) -> Term:
    """``add`` -> ``bias``: the IR's elementwise-add spelling.

    ``bias`` denotes ``tile + bias`` — torch broadcasting, grade
    ROUNDS — so the denotation is an exact add for any broadcastable
    tensor pair.  A ``Const`` operand is declined: it would denote but
    ``Term.cost`` calls ``numel`` on it.
    """
    _check_no_attrs(term)
    if len(term.args) != 2:
        raise UnsupportedOp(
            "add", f"expected 2 args, got {len(term.args)}"
        )
    left, right = term.args
    if isinstance(left, Const) or isinstance(right, Const):
        raise UnsupportedOp(
            "add",
            "scalar (Const) operand: kernel-ir cost model assumes "
            "tensor operands",
        )
    return Term(
        "bias",
        (_tr_arg(left, k_leaves), _tr_arg(right, k_leaves)),
    )


def _transpose(term: Op, k_leaves: Container[str]) -> Term:
    """``transpose(x, dim0, dim1)`` -> kernel-ir ``transpose``.

    The mlopt attrs (``dim0``/``dim1``, ``arg1``/``arg2`` spellings;
    default ``-2``/``-1`` like the torch binding) become positional int
    args — the kernel-ir op takes no attrs.
    """
    if len(term.args) != 1:
        raise UnsupportedOp(
            "transpose", f"expected 1 arg, got {len(term.args)}"
        )
    allowed = {"dim0", "dim1", "arg1", "arg2"}
    extra = set(term.attrs) - allowed
    if extra:
        raise UnsupportedOp(
            "transpose", f"attrs {sorted(extra)} have no counterpart"
        )
    dim0 = _int_attr(term, "dim0", "arg1", default=-2)
    dim1 = _int_attr(term, "dim1", "arg2", default=-1)
    return Term(
        "transpose", (_tr_arg(term.args[0], k_leaves), dim0, dim1)
    )


def _permute(term: Op, k_leaves: Container[str]) -> Term:
    """``permute(x, dims...)`` -> kernel-ir ``permute``.

    The dims live in the attrs (``dim``/``dims``/``order``, any of an
    int, list or tuple); absent, mlopt's convention is reverse-all-axes,
    which needs a known rank.  They become trailing positional int args.
    """
    if len(term.args) != 1:
        raise UnsupportedOp(
            "permute", f"expected 1 arg, got {len(term.args)}"
        )
    allowed = {"dim", "dims", "order"}
    extra = set(term.attrs) - allowed
    if extra:
        raise UnsupportedOp(
            "permute", f"attrs {sorted(extra)} have no counterpart"
        )
    raw = None
    for name in ("dim", "dims", "order"):
        if name in term.attrs:
            raw = term.attrs[name]
            break
    rank = _rank(_leaf_typ(term.args[0]))
    if raw is None:
        if rank is None:
            raise UnsupportedOp(
                "permute",
                "no dims attr and unknown rank — cannot spell "
                "reverse-all-axes",
            )
        dims: tuple[int, ...] = tuple(reversed(range(rank)))
    else:
        if isinstance(raw, int) and not isinstance(raw, bool):
            dims = (raw,)
        elif isinstance(raw, (list, tuple)):
            dims = tuple(raw)
        else:
            raise UnsupportedOp(
                "permute",
                f"dims attr {raw!r} is not an int or int list",
            )
        if not all(
            isinstance(d, int) and not isinstance(d, bool) for d in dims
        ):
            raise UnsupportedOp(
                "permute", f"dims attr {raw!r} has non-int entries"
            )
        if rank is not None and len(dims) != rank:
            raise UnsupportedOp(
                "permute",
                f"{len(dims)} dims for a rank-{rank} operand",
            )
    return Term("permute", (_tr_arg(term.args[0], k_leaves), *dims))


def _linear(term: Op, k_leaves: Container[str]) -> Term:
    """``linear(x[in], W[out, in], b?)`` -> ``bias(dot(Wᵀ, x), b)``.

    ``F.linear`` contracts the input's last axis against the weight's
    dim 1 and emits dim 0 — the transpose of ``dot``'s layout, so the
    weight goes through ``transpose`` first.  Only a rank-1 input maps:
    a batched ``[batch, in]`` input is a matrix-matrix product, not a
    ``dot``.
    """
    _check_no_attrs(term)
    if len(term.args) not in (2, 3):
        raise UnsupportedOp(
            "linear", f"expected 2 or 3 args, got {len(term.args)}"
        )
    x, w = term.args[0], term.args[1]
    tx, tw = _leaf_typ(x), _leaf_typ(w)
    rx, rw = _rank(tx), _rank(tw)
    if rx == 1 and rw == 2:
        kx, kin = _dim(tx, 0), _dim(tw, 1)
        if kx is not None and kin is not None and kx != kin:
            raise UnsupportedOp(
                "linear",
                f"contracted dims differ: x{_shape(tx)} "
                f"vs W{_shape(tw)}",
            )
        weight_t = Term("transpose", (_tr_arg(w, k_leaves), 0, 1))
        out = Term("dot", (weight_t, _tr_arg(x, k_leaves)))
        if len(term.args) == 3:
            return Term("bias", (out, _tr_arg(term.args[2], k_leaves)))
        return out
    raise UnsupportedOp(
        "linear",
        "only the rank-1-input spelling [in] @ [out, in] -> [out] maps "
        f"to dot+transpose; got ranks ({rx}, {rw})",
    )


#: Pointwise ops mapped name-for-name: op -> arity.
_POINTWISE: dict[str, int] = {
    "mul": 2,
    "sub": 2,
    "neg": 1,
    "relu": 1,
    "exp": 1,
    "sigmoid": 1,
    "tanh": 1,
    "gelu": 1,
    "silu": 1,
}


def _pointwise(term: Op, k_leaves: Container[str]) -> Term:
    """``mul``/``sub``/unary activations -> the same-named kernel-ir op.

    All are torch-broadcast pointwise maps; a ``Const`` operand still
    declines (``Term.cost`` needs ``numel``), and attrs decline like
    everywhere else.
    """
    _check_no_attrs(term)
    want = _POINTWISE[term.op]
    if len(term.args) != want:
        raise UnsupportedOp(
            term.op, f"expected {want} args, got {len(term.args)}"
        )
    return Term(term.op, tuple(_tr_arg(a, k_leaves) for a in term.args))


def translate(
    term: Any, *, k_leaves: Container[str] = ()
) -> Term | Leaf | int | float:
    """Lower an ``mlopt_core.term`` term to a ``lib_kernel_ir`` node.

    Returns a :class:`lib_kernel_ir.Term` for mapped ops, a
    :class:`Leaf` for a bare ``Var``/``Param`` root, and raises
    :class:`UnsupportedOp` — naming the op — for everything else.
    ``k_leaves`` names variables a ``split_k`` search should slice on
    the k axis (``Leaf.kind == "k"``); the default leaves everything
    unsliced (``"out"``), which is correct but schedule-blind.
    """
    if isinstance(term, (Var, Param)):
        return _leaf(term, k_leaves)
    if isinstance(term, Const):
        raise UnsupportedOp(
            "const", "a bare scalar is not a tensor program"
        )
    if not isinstance(term, Op):
        raise UnsupportedOp(
            type(term).__name__, "not an mlopt_core.term term"
        )
    if term.op == "matmul":
        return _matmul(term, k_leaves)
    if term.op == "add":
        return _add(term, k_leaves)
    if term.op == "transpose":
        return _transpose(term, k_leaves)
    if term.op == "permute":
        return _permute(term, k_leaves)
    if term.op == "linear":
        return _linear(term, k_leaves)
    if term.op in _POINTWISE:
        return _pointwise(term, k_leaves)
    raise UnsupportedOp(
        term.op,
        "no lib-kernel-ir counterpart — reductions, norms, attention "
        "and layout ops beyond transpose/permute are outside the "
        "vocabulary; see docs/reports/mlopt_ir_kernel_bridge.md",
    )
