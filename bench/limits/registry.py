"""The baseline-adapter contract slice — types only, no registry.

This repo carries one limits baseline (``bench.limits.triton``) and
no limits runner: ``bench.discovery.kernel`` calls
:func:`bench.limits.triton.kernel_builder` directly.  The name→spec
enumeration machinery graphopt's ``bench.limits.registry`` ships
(:func:`specs`/:func:`names`/:func:`for_case`/:func:`availability`/
:func:`build`) had no caller here and is deliberately absent — this
file keeps only the three names the ``triton`` folder's ``SPEC`` /
``build`` contract imports, so the folder stays a drop-in baseline
if a runner lane ever lands.

The contract (mirrors graphopt's ``bench/limits/registry.py``): a
baseline folder exports ``SPEC`` (a :class:`BaselineSpec` row — the
measured record that earned it a seat) and::

    build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Callable

returning the arm to time — or raising :class:`SkipBaseline` with
the reason when it cannot run the case/environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.discovery.cases import DiscoveryCase

__all__ = [
    "BaselineCtx",
    "BaselineSpec",
    "SkipBaseline",
]


class SkipBaseline(Exception):
    """A baseline declining a case — the row records *why*."""


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline's registration — its seat on the roster.

    *cases* is the honest capability statement: the discovery-corpus
    case names the ``build`` accepts (``()`` = every case).
    *requires* names modules that must import for the arm to exist
    (``triton`` — present on CUDA hosts, absent otherwise);
    *needs_cuda* gates CUDA-only kernels; *symbolic* says the arm
    has a dag_cost-pricable form (``False`` = measured-only, the
    fast lane reports it but cannot price it).  *record* is the
    one-line measured record the folder docstring argues at length.
    """

    name: str
    kind: str  # "eager" | "compile" | "order" | "kernel" | "impl" | "carrier"
    title: str = ""
    cases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    needs_cuda: bool = False
    symbolic: bool = True
    note: str = ""
    record: str = ""


@dataclass
class BaselineCtx:
    """Everything a ``build`` may read — the case, its fixture, the run."""

    case: DiscoveryCase
    model: Any  # the spelled "before" module
    feed: Any  # tensor or positional-args tuple
    sr: Any  # the search() result (extracted term inside)
    sink: Any  # TorchSink — verify + lower ports
    source: Any  # TorchSource — the IR exporter
    device: str = "cpu"
    lane: str = "fast"
