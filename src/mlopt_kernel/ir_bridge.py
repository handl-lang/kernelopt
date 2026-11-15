"""Bridge: ``mlopt_core.ir`` terms -> ``lib_kernel_ir`` Terms.

Phase-C spike of the handl-bridge proposal: can a term the mlopt
pipeline extracts (``Optimizer`` phase 3, an ``mlopt_core.ir.Op`` DAG)
be handed to kernelopt's codegen search as a ``lib_kernel_ir.Term``?

The honest answer is *mostly no*, and this module is built so that
"no" is loud.  ``lib_kernel_ir`` is a **closed ten-op** vocabulary
(ADR-0165/0166: ``tile_load``, ``unpack_i4``, ``dequant``, ``dot``,
``split``, ``stage``, ``reduce``, ``store``, ``bias``, ``pipeline``)
oriented at one shape of kernel — the dequantised GEMV.  The mlopt IR
is a ~95-op general tensor graph.  The intersection is two ops:

* ``matmul`` — only the vector-matrix spelling ``matmul(v[k], M[k, n])
  -> [n]`` denotes a ``lib_kernel_ir`` ``dot`` (which is
  ``einsum("kn,k->n")`` — contracted dim *first*).  Every other
  matmul shape declines: the weight-first GEMV ``matmul(W[m, k], v[k])
  -> [m]`` would need a transpose the IR does not have, and general
  ``[m, k] @ [k, n]`` is not a ``dot`` at all.
* ``add`` — ``lib_kernel_ir``'s only elementwise-add spelling is
  ``bias`` (``tile + bias``, torch broadcasting, grade ROUNDS).  The
  denotation is an exact torch add, so ``add`` maps — but only between
  two tensor operands: a ``Const`` operand would denote fine yet crash
  ``Term.cost`` (``_n`` needs ``numel``), so it declines.

Everything else — ``mul``, ``sub``, ``relu``, ``sum``, ``softmax``,
``linear``, ``conv2d``, ``split`` (a *name* collision: mlopt's
``split`` cuts a tensor, kernel-ir's partitions the k axis — declined,
never confused), ``reshape``, ``permute``, … — raises
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
env-side; mlopt preserves dtypes), no transpose/reshape/permute, the
``store``/``out``-buffer epilogue is the caller's job, and ``linear``'s
``[out, in]`` weight layout has no path to ``dot``'s ``[k, n]``.
"""

from __future__ import annotations

from collections.abc import Container
from typing import Any

from lib_kernel_ir import Leaf, Term
from mlopt_core.ir import Const, Op, Param, TensorType, Var

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


def _matmul(term: Op, k_leaves: Container[str]) -> Term:
    """``matmul(v[k], M[k, n]) -> [n]`` is the IR's ``dot`` — and only that.

    ``lib_kernel_ir.dot`` denotes ``einsum("kn,k->n")``: the contracted
    axis is dim 0 of *both* operands, output is dim 1 of the matrix.
    torch's ``matmul(v, M)`` with ``v`` rank-1 and ``M`` rank-2 is the
    same denotation with the args swapped.  The weight-first spelling
    ``matmul(W[m, k], v[k])`` — the common GEMV — would need
    ``W.transpose(0, 1)`` inside the term, and the IR has no transpose:
    declined.  Rank-2 @ rank-2 declines likewise.
    """
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
            sa = (
                ta.shape
                if ta is not None and ta.shape is not None
                else ()
            )
            sb = (
                tb.shape
                if tb is not None and tb.shape is not None
                else ()
            )
            raise UnsupportedOp(
                "matmul",
                f"contracted dims differ: v{tuple(sa)} vs M{tuple(sb)}",
            )
        # dot(matrix, vector): the matrix arg comes first in kernel-ir.
        return Term("dot", (_tr_arg(b, k_leaves), _tr_arg(a, k_leaves)))
    if ra == 2 and rb == 1:
        raise UnsupportedOp(
            "matmul",
            "weight-first GEMV [m,k]@[k] needs a transpose the "
            "kernel IR does not have (dot stores the contracted "
            "dim first) — bind a pre-transposed [k,m] weight "
            "under a fresh leaf name instead",
        )
    raise UnsupportedOp(
        "matmul",
        f"only the [k] @ [k, n] -> [n] GEMV spelling maps to dot; "
        f"got ranks ({ra}, {rb})",
    )


def _add(term: Op, k_leaves: Container[str]) -> Term:
    """``add`` -> ``bias``: the IR's only elementwise-add spelling.

    ``bias`` denotes ``tile + bias`` — torch broadcasting, grade
    ROUNDS — so the denotation is an exact add for any broadcastable
    tensor pair.  A ``Const`` operand is declined: it would denote but
    ``Term.cost`` calls ``numel`` on it.
    """
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


def translate(
    term: Any, *, k_leaves: Container[str] = ()
) -> Term | Leaf | int | float:
    """Lower an ``mlopt_core.ir`` term to a ``lib_kernel_ir`` node.

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
            type(term).__name__, "not an mlopt_core.ir term"
        )
    if term.attrs:
        raise UnsupportedOp(
            term.op,
            f"attrs {sorted(term.attrs)} have no kernel-ir "
            "counterpart (the IR's ops take positional args only)",
        )
    if term.op == "matmul":
        return _matmul(term, k_leaves)
    if term.op == "add":
        return _add(term, k_leaves)
    raise UnsupportedOp(
        term.op,
        "no lib-kernel-ir counterpart — the kernel IR is a closed "
        "ten-op GEMV/dequant vocabulary (tile_load, unpack_i4, "
        "dequant, dot, split, stage, reduce, store, bias, pipeline); "
        "see docs/reports/mlopt_ir_kernel_bridge.md",
    )
