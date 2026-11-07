"""Search over kernel IR terms: gate, bound, then measure.

The order is not negotiable (ADR-0165 rules 4-6):

1. **Enumerate `Term`s.** Structure comes from the IR, not from a config
   grid: `split_k = 4` is four `split` terms with an equal composed
   denotation, so the search ranges over meaning-full bodies.
2. **Prune on the cost bound.** Roofline floor and legality only. The model
   is never allowed to rank two feasible candidates — measured, it
   contradicted deploy ranking on three of four shapes (P10/P12).
3. **Gate against an external incumbent.** Never against a reference we
   wrote: the AWQ nibble-order bug passed exactly such a gate (P21/P22).
4. **Measure.** The timer picks the winner, and a candidate that loses is
   recorded with its reason, not dropped silently.

    result = search(
        operator="qgemv",
        candidates=cands,
        incumbent=lambda: awq_gemm(...),
        bytes_moved=2_180_000,
        bandwidth_gbps=89.1,
    )
    result.routing_row()  # {"operator": ..., "term": ..., "us": ...}

`Candidate`s are injected, so the driver is testable without a GPU and the
lowering (`compile-triton`) is free to arrive later: whatever can produce
a callable and a Term can be searched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from lib_kernel_ir import Cost, Term

__all__ = [
    "Candidate",
    "Rejected",
    "SearchResult",
    "roofline_bound_us",
    "search",
]


@dataclass(frozen=True)
class Candidate:
    """One term, plus the two things needed to judge it."""

    term: Term
    label: str
    #: run it and return its output (the denotation, or the real kernel)
    build: Callable[[], torch.Tensor]
    #: measured per-call time in µs; `None` means "not measurable here"
    measure: Callable[[], float | None]


@dataclass(frozen=True)
class Rejected:
    """A candidate that did not make it, and why."""

    label: str
    reason: str
    detail: str = ""


@dataclass
class SearchResult:
    """The winner, the routing row, and everything that lost."""

    operator: str
    chosen: Candidate | None
    cosine: float
    us: float
    speedup_vs_incumbent: float
    roofline_us: float
    rejected: list[Rejected] = field(default_factory=list)

    @property
    def pct_of_roofline(self) -> float:
        """Return the achieved share of the bandwidth roofline."""
        return 100.0 * self.roofline_us / self.us if self.us else 0.0

    def routing_row(self) -> dict:
        """Return the row a dispatcher reads: operator → kernel + config."""
        return {
            "operator": self.operator,
            "term": self.chosen.label if self.chosen else None,
            "us": round(self.us, 1),
            "cosine": round(self.cosine, 7),
            "speedup_vs_incumbent": round(self.speedup_vs_incumbent, 2),
            "pct_of_roofline": round(self.pct_of_roofline, 1),
        }

    def to_dict(self) -> dict:
        """Return the full result row, routing fields plus rejects."""
        return {
            **self.routing_row(),
            "roofline_us": round(self.roofline_us, 2),
            "rejected": [
                {
                    "label": r.label,
                    "reason": r.reason,
                    "detail": r.detail,
                }
                for r in self.rejected
            ],
        }


def roofline_bound_us(
    bytes_moved: float, bandwidth_gbps: float, overhead_us: float = 0.0
) -> float:
    """Compute the floor a candidate cannot beat: bytes ÷ bandwidth + overhead."""
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")
    return bytes_moved / (bandwidth_gbps * 1e9) * 1e6 + overhead_us


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.detach().flatten().float(),
        b.detach().flatten().float(),
        dim=0,
    ).item()


def search(
    *,
    operator: str,
    candidates: Sequence[Candidate],
    incumbent: Callable[[], torch.Tensor],
    bytes_moved: float,
    bandwidth_gbps: float,
    overhead_us: float = 0.0,
    measure_incumbent: Callable[[], float | None] | None = None,
    tolerance: float = 0.9999,
) -> SearchResult:
    """Gate, bound, measure, pick. See the module docstring for the order."""
    if not candidates:
        raise ValueError("no candidates to search")
    floor = roofline_bound_us(bytes_moved, bandwidth_gbps, overhead_us)
    expected = incumbent()
    incumbent_us = measure_incumbent() if measure_incumbent else 0.0

    rejected: list[Rejected] = []
    feasible: list[tuple[float, Candidate, float]] = []

    for cand in candidates:
        # 2a. the bound: a plan claiming to beat the floor is measuring the
        # wrong thing, so its number is not admissible.
        try:
            term_cost: Cost = cand.term.cost({})
        except Exception:
            term_cost = Cost()
        if term_cost.launches and term_cost.bytes_moved:
            pass  # priced; no ranking is done with it (rule 6)

        # 3. the gate, against the incumbent only.
        try:
            out = cand.build()
        except Exception as exc:
            rejected.append(
                Rejected(
                    cand.label, "failed to run", f"{type(exc).__name__}"
                )
            )
            continue
        cos = _cosine(out, expected)
        if cos < tolerance:
            rejected.append(
                Rejected(
                    cand.label,
                    "gate",
                    f"cosine {cos:.6f} < {tolerance}",
                )
            )
            continue

        # 4. measure.
        us = cand.measure()
        if us is None:
            rejected.append(Rejected(cand.label, "unmeasurable", ""))
            continue
        if us < floor:
            rejected.append(
                Rejected(
                    cand.label,
                    "below the roofline floor",
                    f"{us:.1f} < {floor:.1f} us — mis-measured, not fast",
                )
            )
            continue
        feasible.append((us, cand, cos))

    if not feasible:
        return SearchResult(
            operator=operator,
            chosen=None,
            cosine=0.0,
            us=0.0,
            speedup_vs_incumbent=0.0,
            roofline_us=floor,
            rejected=rejected,
        )

    us, chosen, cos = min(feasible, key=lambda t: t[0])
    for u, cand, _ in feasible:
        if cand is not chosen:
            rejected.append(
                Rejected(
                    cand.label, "slower", f"{u:.1f} us vs {us:.1f} us"
                )
            )
    return SearchResult(
        operator=operator,
        chosen=chosen,
        cosine=cos,
        us=us,
        speedup_vs_incumbent=(incumbent_us / us)
        if incumbent_us
        else 0.0,
        roofline_us=floor,
        rejected=rejected,
    )
