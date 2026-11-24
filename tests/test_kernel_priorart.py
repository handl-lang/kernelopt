"""The kernel prior-art bank — structural contract + cert spots.

``mlopt_kernel.priorart.KERNEL_PRIOR_ART_RULES`` encodes the known
codegen vocabulary (fused-kernel denotations, tiling distributions,
GEMM operand folds, strength reductions) as certified laws — the
derivability floor ``novelty_sieve.known_bank`` saturates against.
These tests pin the bank's shape (unique names, ``PRIOR_ART``-tagged,
held out of ``KERNEL_RULES``), the alpha-sibling exclusion that keeps
the denotation bridges in the premise pool, and two replayable-cert
spot checks.  The full per-law sweep is measured —
``docs/results_prior_art_kernel.md`` (regenerate:
``python -m bench.drivers.screen.priorart_verify``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kit_discovery.verifier import verify_law
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import tags
from mlopt_core.laws.serialize import alpha_key
from mlopt_core.meta import instantiate_pattern

from mlopt_kernel import KERNEL_RULES
from mlopt_kernel.priorart import KERNEL_PRIOR_ART_RULES


def _pa(name: str) -> Rewrite:
    return next(r for r in KERNEL_PRIOR_ART_RULES if r.name == name)


class TestBankShape:
    """The prior-art bank is opt-in premises, never play actions."""

    def test_unique_names(self):
        names = [r.name for r in KERNEL_PRIOR_ART_RULES]
        assert len(names) == len(set(names))

    def test_all_tagged_prior_art(self):
        for r in KERNEL_PRIOR_ART_RULES:
            assert tags.PRIOR_ART in r.tags, r.name

    def test_not_in_kernel_rules(self):
        """Premises never enter the shipped action bank."""
        shipped = {r.name for r in KERNEL_RULES}
        assert shipped.isdisjoint(
            r.name for r in KERNEL_PRIOR_ART_RULES
        )

    def test_no_name_collides_with_substrate(self):
        """``pa_*`` names must not shadow the substrate bank."""
        from mlopt_core.laws import ALL_RULES

        lib = {r.name for r in ALL_RULES}
        assert lib.isdisjoint(r.name for r in KERNEL_PRIOR_ART_RULES)


class TestSiblingPosture:
    """The bridges are expands — never alpha-twins of the folds."""

    def test_silu_bridge_is_not_sibling(self):
        """``kf_silu→spelled`` has a different alpha key than the
        shipped fold, so it survives the premise cut."""
        fold = next(
            r for r in KERNEL_RULES if r.name == "kern_fuse_silu"
        )
        pa = _pa("pa_kf_silu_expand")
        assert alpha_key(pa.lhs, pa.rhs) != alpha_key(
            fold.lhs, fold.rhs
        )

    def test_every_fold_has_a_bridge(self):
        """Each fused-op fold's rhs op gets a denotation expand."""
        bridged = {r.lhs.op for r in KERNEL_PRIOR_ART_RULES if r.lhs.op}
        from mlopt_kernel.laws import (
            FUSED_KERNEL_OPS,
            SCHEDULED_KERNEL_OPS,
        )

        assert bridged >= FUSED_KERNEL_OPS
        assert bridged >= SCHEDULED_KERNEL_OPS


class TestCertSpots:
    """Two representatives carry replayable certs on real sites."""

    def test_silu_bridge_certifies(self):
        x = Var("x", TensorType((4, 8)))
        sub = {"a": x, "b": x}
        lhs = instantiate_pattern(_pa("pa_kf_silu_expand").lhs, sub)
        rhs = instantiate_pattern(_pa("pa_kf_silu_expand").rhs, sub)
        res = verify_law(lhs, rhs, [_pa("pa_kf_silu_expand")])
        assert res.derivable and res.replayable

    def test_div_recip_composes(self):
        """The InstCombine two-step: div-by-const then const-fold
        lands on ``mul(u, Const(0.25))`` — the composition that
        demotes ``kern_div_recip``."""
        u = Var("u", TensorType((4, 4)))
        lhs = Op.make("div", u, Const(4.0))
        rhs = Op.make("mul", u, Const(0.25))
        res = verify_law(
            lhs,
            rhs,
            [_pa("pa_div_const_recip"), _pa("pa_const_fold_recip")],
        )
        assert res.derivable
