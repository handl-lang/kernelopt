"""The cross-domain transfer driver's light coverage.

Episodes on the kernel/search/meta boards are milliseconds; the
``open`` target is oracle-priced, so the suite exercises the driver
on the cheap boards and keeps ``open`` to the pure-function arms.
"""

from __future__ import annotations

from kit_discovery import play

import mlopt_kernel.play
from bench.drivers.policy import transfer
from mlopt_kernel.weights import KERNEL_PLAYER_WEIGHTS

mlopt_kernel.play.register()  # seat the kernel domain explicitly


class TestOverlapMath:
    def test_disjoint_vocabularies_carry_no_mass(self):
        """A weight table whose keys never fire transplants nothing."""
        o = transfer.vocab_overlap({"a:r:01": 1.0}, {"x:m", "bias"})
        assert o["shared"] == [] and o["mass_frac"] == 0.0

    def test_partial_overlap_reports_live_mass(self):
        """Only the colliding keys count toward the live share."""
        o = transfer.vocab_overlap(
            {"a": 1.0, "b": -3.0, "c": 0.0}, {"a", "b", "d"}
        )
        assert o["shared"] == ["a", "b"]
        assert o["mass_frac"] == 1.0

    def test_empty_table_is_zero_not_error(self):
        """An empty source table reports 0 mass, no division error."""
        o = transfer.vocab_overlap({}, {"a"})
        assert o["mass_frac"] == 0.0 and o["src_keys"] == 0


class TestRatio:
    def test_ratio_and_delta(self):
        """Warm/cold is the headline; delta always reports."""
        ratio, delta = transfer.transfer_ratio(0.6, 0.3)
        assert ratio == 2.0 and delta == 0.3

    def test_near_zero_cold_suppresses_ratio(self):
        """A ~0 cold baseline degrades the ratio to None honestly."""
        ratio, delta = transfer.transfer_ratio(0.4, 1e-5)
        assert ratio is None and abs(delta - 0.4) < 1e-4


class TestCompatibilityProbe:
    def test_open_spec_cannot_play_kernel_states(self):
        """The open featurizer reads fields KernelState lacks."""
        spec = play.DOMAINS["open"].spec
        c = transfer.compat(spec, "kernel")
        assert "Error" in c["legal"] or "Error" in c["featurize"]

    def test_search_spec_runs_on_kernel_states(self):
        """The one working cross-application — documented boundary.

        KernelState is a superset of the search board's fields, so
        the search spec enumerates and featurizes kernel states; it
        still cannot see ``stop`` — the compat probe is honest that
        'runs' is not 'transfers'.
        """
        spec = play.DOMAINS["search"].spec
        c = transfer.compat(spec, "kernel")
        assert c["legal"].startswith("ok") and c["featurize"] == "ok"

    def test_kernel_spec_fails_on_search_states(self):
        """The reverse direction dies on ``has_const`` — asymmetric."""
        spec = play.DOMAINS["kernel"].spec
        c = transfer.compat(spec, "search")
        assert "Error" in c["featurize"]


class TestEmittedVocabulary:
    def test_kernel_rows_cover_the_schema_marks(self):
        """Step-0 kernel rows emit the schema's live channels."""
        emitted = transfer.emitted_vocabulary("kernel", seed=0, cases=2)
        assert {"bias", "st:steps", "st:improve", "x:m"} <= emitted

    def test_open_shares_no_names_with_kernel_weights(self):
        """The zero-mass boundary the doc reports — measured."""
        emitted = transfer.emitted_vocabulary("open", seed=0, cases=1)
        assert not (set(KERNEL_PLAYER_WEIGHTS) & emitted)


class TestTransferEval:
    def test_kernel_target_arms_play_episodes(self):
        """The transplant eval runs end-to-end on kernel boards."""
        sources = {
            "meta": transfer.BAKED_SOURCES["meta"],
            "open": transfer.BAKED_SOURCES["open"],
        }
        res = transfer.eval_target(
            "kernel", sources, seed=0, eval_cases=2
        )
        assert "cold-greedy" in res["arms"]
        assert "xfer:meta-greedy" in res["arms"]
        assert "kern-w@search-spec" in res["arms"]
        assert "xfer:kernel-greedy" not in res["arms"]
        for row in res["arms"].values():
            assert "reward" in row and "delta" in row

    def test_search_target_seats_trained_native(self):
        """A prior trained at runtime anchors its own board's table."""
        sources = {"search": {"a:r:02": 1.0}}
        res = transfer.eval_target(
            "search", sources, seed=0, eval_cases=2
        )
        assert "native-greedy" in res["arms"]
        assert "xfer:search-greedy" not in res["arms"]
