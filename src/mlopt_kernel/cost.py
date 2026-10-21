"""The kernel cost model — launch- and op-weight-dominated pricing.

The codegen laws' value lives here.  Two things price a kernel
program: *launches* (every op the spelled program emits is a kernel
dispatch; a fused op is exactly one) and *op weight* (a divide or a
transcendental is several times the throughput cost of a
multiply/add on device — the reason strength reduction and
reciprocal folding pay).  Layout ops are the exception: a ``split``
/``select``/``contiguous``/``detach_`` (or a reshape-family view)
emits no kernel at all — they cost zero launches, so the model
never charges a real carrier term for its slicing.

``kf_*`` fused ops weigh one launch — the sink emits one compiled
kernel per fused op.  The model is additive per node, so
``dag_cost``'s subtractive decomposition and ``extract_best``'s
local-cost recovery both handle it exactly like ``count_cost``.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ir import Op

from mlopt_kernel.laws import FUSED_KERNEL_OPS, SCHEDULED_KERNEL_OPS

__all__ = ["kernel_cost"]

#: One kernel launch — the unit axis.
_LAUNCH = 1.0

#: Per-op throughput weight (in launches) — the expensive elementary
#: ops a device kernel replaces with cheaper arithmetic.  Rough
#: vendor-agnostic figures; the point is the *ordering*, not the
#: calibration.
_OP_WEIGHT = {
    "pow": 4.0,
    "div": 2.0,
    "rsqrt": 2.0,
    "sqrt": 2.0,
    "exp": 2.0,
    "log": 2.0,
    "tanh": 2.0,
    "sigmoid": 2.0,
    "gelu": 2.0,
    "softmax": 2.0,
    "erf": 2.0,
}

#: Zero-launch ops — the view/layout vocabulary.  Every op here is a
#: metadata-only transform on device (a stride/shape fixup or an
#: autograd no-op); the sink binds them to views, so the cost model
#: prices them free.  ``concat`` is deliberately absent — ``cat``
#: materialises a copy, a real launch.
_VIEW_OPS = frozenset(
    {
        "contiguous",
        "detach",
        "detach_",
        "flatten",
        "permute",
        "reshape",
        "select",
        "split",
        "squeeze",
        "transpose",
        "unsqueeze",
    }
)

#: Zero-launch schedule markers — ``om_chunk_scan`` emits no kernel
#: of its own: the node marks a schedule boundary whose launches are
#: the batched GEMMs billed at its arm operands (and priced by their
#: subtrees).  Below the ``concat`` it replaces (a materialising
#: copy, one launch), so a marked concat spine strictly wins per
#: node — the nested-binary fold extracts whole, not piecemeal.
_ZERO_OPS = _VIEW_OPS | SCHEDULED_KERNEL_OPS


def kernel_cost(term: Any, memo: dict | None = None) -> float:
    """Cost = launches + op weights; a fused op is one launch.

    Follows the shipped cost-fn convention — ``(term, memo=None) ->
    float``, recursive, content-keyed memo — so ``dag_cost`` /
    ``extract_best`` consume it like ``count_cost``.
    """
    memo = {} if memo is None else memo
    ck = ("kc", term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        w = (
            0.0
            if term.op in _ZERO_OPS
            else _LAUNCH
            if term.op in FUSED_KERNEL_OPS
            else _OP_WEIGHT.get(term.op, _LAUNCH)
        )
        v = w + sum(kernel_cost(a, memo) for a in term.args)
        memo[ck] = float(v)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0
