"""The baseline-adapter contract — re-exported from ``benchkit.baselines``.

The three names are the shared contract (canonical home:
``handl-lang/gamekit``'s ``packages/benchkit`` — the same types
graphopt's ``bench.limits.registry`` re-exports).  This repo carries
one limits baseline (``bench.limits.triton``) and no limits runner:
``bench.discovery.kernel`` calls
:func:`bench.limits.triton.kernel_builder` directly.  The name→spec
enumeration machinery graphopt's ``bench.limits.registry`` ships
(:func:`specs`/:func:`names`/:func:`for_case`/:func:`availability`/
:func:`build`) had no caller here and is deliberately absent — the
``triton`` folder's ``SPEC`` / ``build`` contract imports only these
three names, so the folder stays a drop-in baseline if a runner lane
ever lands.

The contract: a baseline folder exports ``SPEC`` (a
:class:`BaselineSpec` row — the measured record that earned it a
seat) and::

    build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Callable

returning the arm to time — or raising :class:`SkipBaseline` with
the reason when it cannot run the case/environment.
"""

from __future__ import annotations

from benchkit.baselines import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)

__all__ = [
    "BaselineCtx",
    "BaselineSpec",
    "SkipBaseline",
]
