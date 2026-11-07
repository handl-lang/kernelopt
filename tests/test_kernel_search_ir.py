"""The search driver: gate first, bound through, measure last.

Every test here runs on CPU with injected candidates, so the *order* of
operations is what is being tested — not a kernel. The ordering is the
whole content of the design: a candidate that is fastest but wrong must
lose, and a number below the roofline floor must be treated as a broken
measurement rather than a win (P21: the same kernel measured 4.7x apart on
two clocks).
"""

from __future__ import annotations

import pytest
import torch

from lib_kernel_ir import Cost, qgemv
from mlopt_kernel.search import (
    Candidate,
    Rejected,
    SearchResult,
    roofline_bound_us,
    search,
)

ROOF_ARGS = {"bytes_moved": 2_180_000.0, "bandwidth_gbps": 89.1}


class StubTerm:
    """A term whose cost is whatever the test needs it to be."""

    def __init__(self, launches: int = 1, bytes_moved: int = 0) -> None:
        self._cost = Cost(bytes_moved=bytes_moved, launches=launches)

    def cost(self, _env: dict) -> Cost:
        return self._cost


def cand(
    label: str,
    *,
    cos_offset: float = 0.0,
    us: float | None = 100.0,
    term=None,
    raises: bool = False,
) -> Candidate:
    expected = torch.arange(1, 33, dtype=torch.float32)

    def build() -> torch.Tensor:
        if raises:
            raise RuntimeError("boom")
        return expected + cos_offset

    def measure() -> float | None:
        return us

    return Candidate(
        term=term or StubTerm(),
        label=label,
        build=build,
        measure=measure,
    )


def run(cands, **kwargs) -> SearchResult:
    base = dict(
        operator="qgemv",
        candidates=cands,
        incumbent=lambda: torch.arange(1, 33, dtype=torch.float32),
        **ROOF_ARGS,
    )
    base.update(kwargs)
    return search(**base)


def test_roofline_bound_is_bytes_over_bandwidth() -> None:
    us = roofline_bound_us(2_180_000, 89.1)
    assert us == pytest.approx(24.47, abs=0.05)
    assert roofline_bound_us(
        2_180_000, 89.1, overhead_us=10
    ) == pytest.approx(us + 10)
    with pytest.raises(ValueError, match="bandwidth"):
        roofline_bound_us(1, 0)


def test_fastest_gated_candidate_wins() -> None:
    result = run([cand("slow", us=90.0), cand("fast", us=40.0)])
    assert result.chosen is not None
    assert result.chosen.label == "fast"
    assert result.us == 40.0
    assert any(
        r.label == "slow" and r.reason == "slower"
        for r in result.rejected
    )


def test_wrong_output_loses_even_when_fastest() -> None:
    """P22: agreement with the incumbent is the entry ticket, not speed."""
    result = run([cand("wrong", cos_offset=1e6, us=1.0)])
    assert result.chosen is None
    assert result.rejected[0].reason == "gate"
    assert "cosine" in result.rejected[0].detail


def test_below_the_floor_is_a_broken_measurement_not_a_win() -> None:
    result = run([cand("impossible", us=1.0), cand("honest", us=40.0)])
    assert result.chosen is not None
    assert result.chosen.label == "honest"
    reasons = {r.label: r.reason for r in result.rejected}
    assert reasons["impossible"] == "below the roofline floor"


def test_unmeasurable_candidate_is_recorded() -> None:
    result = run([cand("no-timer", us=None), cand("ok", us=50.0)])
    assert result.chosen is not None and result.chosen.label == "ok"
    assert any(r.reason == "unmeasurable" for r in result.rejected)


def test_a_candidate_that_raises_does_not_stop_the_search() -> None:
    result = run([cand("boom", raises=True), cand("ok", us=50.0)])
    assert result.chosen is not None and result.chosen.label == "ok"
    assert any(r.reason == "failed to run" for r in result.rejected)


def test_cost_never_ranks_candidates() -> None:
    """Rule 6: an expensive term may still be the fastest and must win."""
    expensive = cand("expensive", us=30.0, term=StubTerm(launches=9))
    cheap = cand("cheap", us=80.0, term=StubTerm(launches=1))
    assert run([expensive, cheap]).chosen.label == "expensive"


def test_speedup_uses_the_incumbent_measured_in_the_same_run() -> None:
    result = run(
        [cand("fast", us=40.0)], measure_incumbent=lambda: 160.0
    )
    assert result.speedup_vs_incumbent == pytest.approx(4.0)
    assert result.routing_row()["speedup_vs_incumbent"] == 4.0


def test_routing_row_is_what_a_dispatcher_reads() -> None:
    row = run([cand("fast", us=40.0)]).routing_row()
    assert row["operator"] == "qgemv"
    assert row["term"] == "fast"
    assert row["us"] == 40.0
    assert row["pct_of_roofline"] == pytest.approx(
        100 * roofline_bound_us(2_180_000, 89.1) / 40.0, abs=0.1
    )


def test_real_terms_carry_their_cost_through_the_driver() -> None:
    """A real `qgemv` term is accepted and priced without ranking."""
    term = qgemv(k=512, split=4)
    result = run([cand("split4", us=60.0, term=term)])
    assert result.chosen is not None
    assert result.to_dict()["term"] == "split4"


def test_empty_candidate_list_is_an_error() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        run([])


def test_rejection_serialises() -> None:
    r = Rejected("x", "gate", "cosine 0.21")
    assert {
        "label": r.label,
        "reason": r.reason,
        "detail": r.detail,
    } == {
        "label": "x",
        "reason": "gate",
        "detail": "cosine 0.21",
    }
