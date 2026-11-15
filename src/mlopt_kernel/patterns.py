# ruff: noqa: RUF002, RUF003 — math notation (×, –, —) in docstrings.
"""``mlopt_kernel.patterns`` — whole-subgraph lowering patterns.

The scan.py precedent, generalised: :mod:`mlopt_kernel.scan` detects
one schedule-level pattern (``om_chunk_scan`` / ``concat``-of-chunks)
and hands the executor a batched :class:`ScanRunner`; this module is
the registry of *structural* patterns — subgraph shapes the spelled
term already carries, detected at lowering time with no law minting
a marker and no cost-model change.  A matched site lowers to a fused
template forward (one ``torch.compile`` region over the composite),
verify-first through :meth:`KernelSink.verify` exactly like the
``kf_*`` fused ops.

First pattern — the **residual projection**::

    add(x, linear(y, w[, b]))     /     add(x, matmul(y, w))

— the transformer decoder layer's most frequent whole-subgraph
(the r4 lifted census: ``linear``×10 and the residual ``add``s that
close every sub-layer — q/k/v/o projections, the MLP's fc1/fc2 pair,
the encoder-decoder attention outs).  Spelled, the site dispatches a
GEMM then a pointwise add; the fused template compiles ``x +
proj(y)`` as one region so the residual folds into the GEMM's
epilogue instead of launching a second kernel.

Detection is deliberately conservative — structure only, no shape
reasoning (the spelled add's broadcast semantics are the spec, so
any shapes the spelled program accepts, the composite accepts):

* the head must be a binary ``add`` with no attrs;
* the projection child must be ``linear`` (2–3 args) or ``matmul``
  (2 args), attr-free — the spelled binding's own arity;
* either operand order matches (``x + proj`` and ``proj + x`` are
  the same site); when *both* children are projections the second
  operand is the fused one and the first evaluates through the
  normal channel — still one fewer launch, still exact.

Anything else returns ``None`` and the executor keeps the spelled
per-op dispatch — a drop-in, never a silent drop.  Because no new op
name is minted, ``kernel_cost``/``supported_ops`` are untouched:
this is a delivery-time schedule, not a search move.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ir import Op

__all__ = ["ResidualProjRunner", "match_residual_proj"]


def _torch() -> Any:
    """Return torch lazily — same layering posture as the sink."""
    import torch

    return torch


def _proj_args(t: Op) -> tuple | None:
    """Return the projection operand terms, or ``None``.

    ``linear`` spells ``F.linear(x, w[, b])`` — 2 or 3 args, attr-free
    (the binding's own contract); ``matmul`` spells the raw
    contraction — 2 args.  Anything else is not a projection site.
    """
    if t.attrs:
        return None
    if t.op == "linear" and len(t.args) in (2, 3):
        return t.args
    if t.op == "matmul" and len(t.args) == 2:
        return t.args
    return None


def _residual_add(t: Op) -> tuple | None:
    """Split ``add(x, proj)`` in either operand order, or ``None``."""
    if t.op != "add" or len(t.args) != 2 or t.attrs:
        return None
    a, b = t.args
    for residual, proj in ((a, b), (b, a)):
        if (
            isinstance(proj, Op)
            and (pargs := _proj_args(proj)) is not None
        ):
            return (residual, proj.op, pargs)
    return None


def match_residual_proj(term: Any) -> dict | None:
    """Detect the residual-projection site in *term*, or ``None``.

    Returns ``{"kind": "linear" | "matmul", "residual": <term>,
    "proj_args": <tuple>}`` — the pieces :class:`ResidualProjRunner`
    needs.  *term* itself must be the ``add`` node: the executor
    probes every op, so non-``add`` roots decline on the first check.
    """
    if not isinstance(term, Op):
        return None
    m = _residual_add(term)
    if m is None:
        return None
    residual, kind, pargs = m
    return {"kind": kind, "residual": residual, "proj_args": pargs}


# ---------------------------------------------------------------------------
#  The fused templates — one compiled region per site
# ---------------------------------------------------------------------------


def _res_linear(x: Any, y: Any, w: Any, b: Any = None) -> Any:
    """``x + F.linear(y, w, b)`` — the residual-projection composite."""
    torch = _torch()
    return x + torch.nn.functional.linear(y, w, b)


def _res_matmul(x: Any, y: Any, w: Any) -> Any:
    """``x + y @ w`` — the matmul-spelled residual projection."""
    torch = _torch()
    return x + torch.matmul(y, w)


_TEMPLATES = {"linear": _res_linear, "matmul": _res_matmul}


class ResidualProjRunner:
    """The fused forward for one residual-projection site.

    Holds the plan and a lazily ``torch.compile``d template callable
    (cached per runner — the executor builds one runner per matched
    node).  ``run(ev)`` is the whole forward: *ev* is the executor's
    operand channel — the residual and every projection operand
    evaluate through it, so live subterms hit the per-call memo and
    const subtrees the lowering-time static cache exactly as the
    spelled dispatch would.
    """

    def __init__(self, plan: dict) -> None:
        """Bind the plan; the template compiles on first forward."""
        self._plan = plan
        self._compiled: Any = None

    def _fn(self) -> Any:
        """Return the compiled template for the plan's kind."""
        if self._compiled is None:
            self._compiled = _torch().compile(
                _TEMPLATES[self._plan["kind"]], fullgraph=False
            )
        return self._compiled

    def run(self, ev: Any) -> Any:
        """Evaluate the site as ``residual + proj(args)`` — one region."""
        p = self._plan
        args = [ev(t) for t in p["proj_args"]]
        if p["kind"] == "linear" and len(args) == 2:
            args.append(None)  # F.linear's optional bias slot
        return self._fn()(ev(p["residual"]), *args)
