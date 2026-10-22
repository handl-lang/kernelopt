"""The kernel novelty sieve — the three-gate filter's contract.

``bench/drivers/screen/novelty_sieve.py`` decides whether a shipped
``KERNEL_RULES`` law is worth a human's eyes: certified (its own rule
carries a replayable certificate on a real instance), paying (a
registered ``bench.discovery.kernel`` case's certified extraction
under the full kernel bank lands strictly below the best the rest of
the bank reaches — *and* the derivation uses the candidate), and
underivable (the kernel bank plus the kernel prior-art bank plus the
shared substrate surface — minus the candidate's alpha-siblings —
does not already prove it).

These tests pin the gate semantics on laws whose answers are known a
priori:

* the shipped ``kern_square_strength`` demotes to ``known`` — the
  substrate bank proves ``pow(x,2) → x·x`` via the very
  ``pow_to_square`` + ``square_expand`` derivation the law declares;
* ``kern_fuse_chunk_scan_3`` demotes to ``known`` — the prior-art
  denotation ``pa_om_chunk_scan_expand_3`` (a scheduled scan block
  IS its spelled concat) proves the fold instance directly;
* a fused-op mint (``mul(a,b) → kf_sieve``) is underivable by
  construction — no premise emits the (unbound) ``kf_sieve`` op —
  and reaches ``review`` off a stubbed measured win;
* a false codegen law (``mul(x,x) → add(x,x)``) is refuted by the
  fp64 ``KernelSink`` oracle before the gates even run;
* a rule whose guard accepts nothing reports ``no-instance``;
* the real measured gate runs: ``kern_fuse_silu`` beats its own
  case's floor against the bank=``KERNEL_RULES`` floor, and
  ``kern_fuse_addmul`` reports ``unpaid`` — its sites are absorbed
  by the gen-2 residue folds.

The cert/derivability gates run on the real machinery —
``verify_law``, ``alpha_key``, the real bank.  The measured gate is
exercised both through the ``measurer`` seam and for real on the
registered corpus (the driver-side full run lives in
``docs/results_kernel_sieve.md``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.ir import Op
from mlopt_core.laws import R
from mlopt_core.laws.serialize import alpha_key

from bench.drivers.screen import novelty_sieve as ns
from mlopt_kernel import KERNEL_RULES


def _rule(name):
    """Look a bank rule up by name — indices churn as the bank grows."""
    return next(r for r in KERNEL_RULES if r.name == name)


# -- the synthetic laws ---------------------------------------------------
#
# KF_SIEVE: mul(a,b) -> kf_sieve(a,b).  True-by-construction (the mint
# defines the op) and underivable: nothing in the kernel bank or the
# substrate emits a ``kf_sieve`` op.  The truth oracle skips it (the
# sink has no binding — inconclusive, not a refute).
KF_SIEVE = R(
    "sieve_kf_mint",
    Op.make("mul", "a", "b"),
    Op.make("kf_sieve", "a", "b"),
    law="mint a fused op — the candidate-pool shape",
)
#: The same law under metavariable renaming — an alpha-sibling.
KF_SIEVE_RENAMED = R(
    "sieve_kf_mint_renamed",
    Op.make("mul", "p", "q"),
    Op.make("kf_sieve", "p", "q"),
    law="renamed spelling",
)
# BAD_EQ: mul(x,x) -> add(x,x).  False on generic feeds — both sides
# lower through the sink, so the fp64 oracle measures the inequality.
BAD_EQ = R(
    "sieve_bad_eq",
    Op.make("mul", "x", "x"),
    Op.make("add", "x", "x"),
    law="a false codegen law",
)


@pytest.fixture(scope="module")
def bank() -> list:
    """The real known bank — derivability is tested against it."""
    return ns.known_bank()


@pytest.fixture(scope="module")
def ctxs() -> list:
    """The real kernel corpus — pure spelled IR, cheap to build."""
    return ns.case_contexts()


def _won(_c, _ctxs, **_kw):
    """A measured-gate stub reporting one attributed win."""
    return True, [
        ns.BeatRow(
            case="synthetic",
            certified=True,
            verified=True,
            fired=1,
            used=True,
            cost_before=10.0,
            cost_after=4.0,
            floor=9.0,
            beat=True,
        )
    ]


def _lost(_c, _ctxs, **_kw):
    """A measured-gate stub reporting no win."""
    return False, [
        ns.BeatRow(
            case="synthetic",
            certified=True,
            verified=True,
            fired=1,
            used=True,
            cost_before=10.0,
            cost_after=12.0,
            floor=9.0,
            beat=False,
        )
    ]


class TestSiblingExclusion:
    """The premise bank must not contain the candidate's own law."""

    def test_alpha_key_matches_across_renaming(self):
        assert alpha_key(KF_SIEVE.lhs, KF_SIEVE.rhs) == alpha_key(
            KF_SIEVE_RENAMED.lhs, KF_SIEVE_RENAMED.rhs
        )

    def test_candidate_in_bank_is_excluded(self, bank):
        premises, sibs = ns._premises(KF_SIEVE, [*bank, KF_SIEVE])
        assert "sieve_kf_mint" in sibs
        assert all(r.name != "sieve_kf_mint" for r in premises)

    def test_renamed_sibling_is_excluded(self, bank):
        premises, sibs = ns._premises(
            KF_SIEVE, [*bank, KF_SIEVE_RENAMED]
        )
        assert "sieve_kf_mint_renamed" in sibs
        assert all(r.name != "sieve_kf_mint_renamed" for r in premises)

    def test_bank_covers_kernel_and_substrate(self, bank):
        """The known bank is kernel rules U the substrate surface."""
        names = {r.name for r in bank}
        assert {r.name for r in KERNEL_RULES} <= names
        # the substrate members kern_square_strength cites
        assert {"pow_to_square", "square_expand"} <= names


class TestGates:
    """Per-gate verdicts on laws whose status is known a priori."""

    def test_square_strength_is_known(self, bank, ctxs):
        """The self-declared substrate derivation demotes the law."""
        row = ns.sieve(
            _rule("kern_square_strength"),
            bank,
            ctxs=ctxs,
            measure=False,
        )
        assert row.cert
        assert row.derivable is True
        assert row.verdict == "known"
        assert {"pow_to_square", "square_expand"} <= set(
            row.witness_rules
        )

    def test_minted_fused_op_reaches_review(self, bank, ctxs):
        row = ns.sieve(KF_SIEVE, bank, ctxs=ctxs, measurer=_won)
        assert row.cert
        # kf_sieve has no sink binding — the oracle skips, not refutes
        assert row.truth == "skip"
        assert row.derivable is False
        assert row.beat is True
        assert row.verdict == "review"

    def test_underivable_without_pay_is_unpaid(self, bank, ctxs):
        row = ns.sieve(KF_SIEVE, bank, ctxs=ctxs, measurer=_lost)
        assert row.derivable is False
        assert row.beat is False
        assert row.verdict == "unpaid"

    def test_underivable_unexercised_is_unmeasured(self, bank):
        row = ns.sieve(KF_SIEVE, bank, ctxs=[], measure=True)
        assert row.derivable is False
        assert row.beat is None
        assert row.verdict == "unmeasured"

    def test_false_law_is_refuted(self, bank, ctxs):
        row = ns.sieve(BAD_EQ, bank, ctxs=ctxs, measure=False)
        assert row.verdict == "refuted"
        assert row.truth == "unequal"

    def test_no_instance_law(self, bank):
        """A rule whose guard vetoes every binding → no-instance."""
        hopeless = R(
            "sieve_hopeless",
            Op.make("add", "x", "y"),
            "x",
            check=lambda _bound: False,
        )
        row = ns.sieve(hopeless, bank, ctxs=[], measure=False)
        assert row.verdict == "no-instance"


class TestMeasuredGateReal:
    """The beat gate on the real corpus — no stubs."""

    def test_silu_beats_its_case_floor(self, ctxs):
        row = ns.sieve(
            _rule("kern_fuse_silu"), ctxs=ctxs, bank=KERNEL_RULES
        )
        assert row.verdict == "review"
        wins = {b.case for b in row.beats if b.beat}
        assert "kernel_silu" in wins
        win = next(b for b in row.beats if b.beat)
        assert win.cost_after < win.floor
        assert win.used and win.certified

    def test_chunk_scan_beats_its_case_floor(self, ctxs):
        """The scheduled fold pays below the rest-of-bank extraction."""
        row = ns.sieve(
            _rule("kern_fuse_chunk_scan"),
            ctxs=ctxs,
            bank=KERNEL_RULES,
        )
        assert row.verdict == "review"
        win = next(
            b for b in row.beats if b.case == "kernel_chunk_scan"
        )
        assert win.beat and win.verified and win.used

    def test_chunk_scan_nary_is_known(self, bank, ctxs):
        """The flat-3 member: the chunk-scan denotation bridge makes
        the fold a theorem of the bank (``pa_om_chunk_scan_expand_3``
        — a scheduled ``om_chunk_scan`` IS the concat it schedules),
        so it demotes to ``known`` before the beat gate runs."""
        row = ns.sieve(
            _rule("kern_fuse_chunk_scan_3"),
            bank,
            ctxs=ctxs,
            measure=True,
        )
        assert row.instance_src == "corpus:xcorpus:scan_3"
        assert row.cert and row.derivable is True
        assert "pa_om_chunk_scan_expand_3" in row.witness_rules
        assert row.verdict == "known"

    def test_addmul_is_unpaid_against_residue_folds(self, ctxs):
        """``kern_fuse_addmul`` never uniquely wins — its sites are
        all absorbed by the gen-2 residue folds (addmul_tanh /
        addmul2 / mm_scale_bias), so the floor already prices them."""
        row = ns.sieve(
            _rule("kern_fuse_addmul"), ctxs=ctxs, bank=KERNEL_RULES
        )
        assert row.verdict == "unpaid"
        assert row.derivable is False
        assert all(
            not b.used or b.cost_after >= b.floor for b in row.beats
        )

    def test_floor_arm_omits_candidate(self, ctxs):
        """The floor is the rest-of-bank extraction, ≤ spelled."""
        row = ns._measure_case(
            _rule("kern_fuse_silu"),
            next(c for c in ctxs if c.name == "kernel_silu"),
        )
        assert row.floor is not None
        assert row.floor <= row.cost_before
        assert row.cost_after < row.floor


class TestRowShape:
    """The record is JSON-shaped for the queue doc."""

    def test_to_dict_keys(self, bank, ctxs):
        row = ns.sieve(KF_SIEVE, bank, ctxs=ctxs, measurer=_won)
        d = row.to_dict()
        for k in ("cert", "derivable", "beat", "verdict", "name"):
            assert k in d
        assert d["verdict"] == "review"
        assert d["family"] == "other"

    def test_render_has_queue_section(self, bank, ctxs):
        rows = [
            ns.sieve(
                _rule("kern_square_strength"),
                bank,
                ctxs=ctxs,
                measure=False,
            ),
            ns.sieve(KF_SIEVE, bank, ctxs=ctxs, measurer=_won),
        ]
        md = ns.render_markdown(rows)
        assert "## Review queue" in md
        assert "`sieve_kf_mint`" in md
        assert "known" in md


class TestMeasureGateStructure:
    """The measured gate's relevance filter and row shape."""

    def test_no_contexts_means_unmeasured(self):
        beat, rows = ns.measure_gate(KF_SIEVE, [])
        assert beat is None
        assert rows == []

    def test_xcorpus_never_measures(self, ctxs):
        """xcorpus sites find instances; they are not benchmarks."""
        xc = [c for c in ctxs if not c.registered]
        assert xc and all(c.name.startswith("xcorpus:") for c in xc)
        beat, rows = ns.measure_gate(
            _rule("kern_fuse_chunk_scan_4"), ctxs
        )
        assert beat is None and rows == []

    def test_beatrow_defaults(self):
        r = ns.BeatRow(case="c")
        assert not r.beat and not r.certified and r.floor is None
        assert r.verified is None
