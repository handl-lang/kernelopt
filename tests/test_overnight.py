"""The kernel overnight loop — episodes -> mint -> admit -> classify.

``bench/drivers/overnight.py`` runs the discovery funnel on the
registered ``bench.discovery.kernel`` corpus: each episode's arena
pool is minted from the case's own term (corpus-census proposals +
``meta.synthesize_rules`` pair-composites over ``KERNEL_RULES`` and
the substrate's guarded parents), evaluated through the sink
(eager fp64 verify + ``kernel_cost`` delta), admitted through the
kernel gauntlet (oracle -> pay -> sweep -> cert), and classified by
the novelty sieve's gates.

These tests pin the contract on cases whose answers are known a
priori:

* the mint emits the expected composites — ``kernel_powscale``'s
  ``pow(x,2)·y`` chain yields ``mul(pow(x,2),y) -> kf_mul3(x,x,y)``
  (the ``square -> mul3`` fold compressed into one law);
* the emit gate drops mints that bind nothing or cannot denote;
* the arena verdicts are the open game's tiers — a derivable,
  cost-lowering composite reads ``useful``, a bank alpha-duplicate
  ``known``, a measured-false mint ``false``;
* the gauntlet admits the composite (oracle equal, verified pay,
  swept sites equal, replayable cert) and refuses a false mint at
  ``oracle``;
* the sieve classifies the composite ``known`` — the bank derives
  the compressed chain stepwise — and ``recognize:softmax``
  ``unpaid`` (underivable, but the bank's ``kf_softmax`` floor
  already owns the win);
* the ledger replays done cells on resume and restarts loudly on a
  protocol mismatch.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from kit_discovery.proposal import Candidate
from mlopt_core.ir import Const, Op
from mlopt_core.laws import R
from mlopt_core.laws.serialize import alpha_key

from bench.drivers import overnight as on
from bench.drivers.screen import novelty_sieve as ns
from mlopt_kernel import KERNEL_RULES


def _rule(name):
    """Look a bank rule up by name — indices churn as the bank grows."""
    return next(r for r in KERNEL_RULES if r.name == name)


#: A composite shape the mint produces on kernel_powscale —
#: ``mul(pow(x,2), y) -> kf_mul3(x, x, y)``: true (the chain is a
#: library theorem), paying (fires the fold in one step), derivable
#: (the bank replays square-then-mul3).
POW_MUL3 = R(
    "test_syn_square_mul3",
    Op.make("mul", Op.make("pow", "x", Const(2)), "y"),
    Op.make("kf_mul3", "x", "x", "y"),
    law="the powscale composite, spelled by hand",
)
#: A false codegen mint — both sides lower through the sink, so the
#: fp64 oracle measures the inequality.
BAD_EQ = R(
    "test_bad_eq",
    Op.make("mul", "x", "x"),
    Op.make("add", "x", "x"),
    law="a false codegen law",
)
#: An unbound-shaped mint — nothing in the corpus has a `qr` op.
UNBOUND = R(
    "test_unbound",
    Op.make("mul", Op.make("qr", "x"), "y"),
    Op.make("kf_mul3", "x", "x", "y"),
    law="binds nothing in the kernel corpus",
)
#: A mint whose guard vetoes every binding — the no-instance shape.
HOPELESS = R(
    "test_hopeless",
    Op.make("add", "x", "y"),
    "x",
    check=lambda _bound: False,
)


def _case(name, **kw):
    """A case dict as ``gen_cases`` builds it (page/compose knobs)."""
    return {
        "kernel": name,
        "category": "speed",
        "page": kw.get("page", 8),
        "compose": kw.get("compose", 1),
    }


@pytest.fixture(scope="module")
def ctxs() -> list:
    """The real kernel corpus contexts — pure spelled IR, cheap."""
    return ns.case_contexts()


@pytest.fixture(scope="module")
def bank() -> list:
    """The real known bank — derivability is tested against it."""
    return ns.known_bank()


class TestLedger:
    """The append-only, protocol-gated cell store."""

    def test_record_reopen_replays(self, tmp_path):
        p = tmp_path / "l.jsonl"
        led = on._Ledger.open(p, on._manifest({"a": 1}))
        led.record(("episode", 0), {"evals": [{"name": "x"}]})
        led.record(("candidate", "x"), {"admitted": True})
        again = on._Ledger.open(p, on._manifest({"a": 1}))
        assert again.done(("episode", 0))
        assert again.get(("candidate", "x"))["admitted"] is True

    def test_protocol_mismatch_restarts(self, tmp_path):
        p = tmp_path / "l.jsonl"
        led = on._Ledger.open(p, on._manifest({"a": 1}))
        led.record(("episode", 0), {"evals": []})
        fresh = on._Ledger.open(p, on._manifest({"a": 2}))
        assert not fresh.done(("episode", 0))
        # the manifest is the first line; the old cell is gone
        rows = [
            json.loads(ln) for ln in p.read_text().splitlines() if ln
        ]
        assert rows[0]["_manifest"] and len(rows) == 1

    def test_torn_tail_tolerated(self, tmp_path):
        p = tmp_path / "l.jsonl"
        led = on._Ledger.open(p, on._manifest({"a": 1}))
        led.record(("episode", 0), {"evals": []})
        with p.open("a") as f:
            f.write('{"key": ["candidate",')
        again = on._Ledger.open(p, on._manifest({"a": 1}))
        assert again.done(("episode", 0))


class TestMint:
    """The pool the arena builds from a case's own term."""

    def test_composites_minted_on_powscale(self):
        arena = on._KernelArena(_case("kernel_powscale"), fuel=64)
        names = [c.as_rule().name for c in arena.pool]
        assert any(
            "square_strength" in n and "mul3" in n for n in names
        ), names
        comp = next(c for c in arena.pool if "mul3" in c.as_rule().name)
        m = arena.meta[arena.pool.index(comp)]
        assert m["family"] == "synth-composite"
        assert m["sources"] == [
            "kern_square_strength",
            "kern_fuse_mul3",
        ]

    def test_mints_bind_and_denote(self):
        """Every pooled rule passes the emit gate on the case term."""
        arena = on._KernelArena(_case("kernel_sdpa"), fuel=64)
        subs = on._op_subterms(arena.term)
        assert arena.pool
        for c in arena.pool:
            assert on._emittable(c.as_rule(), subs) is not None

    def test_emit_gate_drops_unbound(self):
        arena = on._KernelArena(_case("kernel_powscale"), fuel=64)
        subs = on._op_subterms(arena.term)
        assert on._emittable(UNBOUND, subs) is None
        bad_rhs = R(
            "test_unbound_rhs",
            Op.make("mul", "x", "y"),
            Op.make("add", "x", "zz"),
        )
        assert on._emittable(bad_rhs, subs) is None

    def test_census_mint_is_corpus_wide(self):
        """The shared census pool binds the cases that carry its
        sites — ``kernel_decayed_pair`` pools the unsqueeze naturals
        the flat scan spelling offers."""
        arena = on._KernelArena(
            _case("kernel_decayed_pair", compose=0), fuel=64
        )
        fams = {arena.meta[i]["family"] for i in arena.meta}
        assert "census-mixed-view" in fams, fams


class TestArenaEval:
    """The kernel candidate evaluation behind ``evaluate``."""

    def _cand(self, rule, name=None):
        return Candidate(
            strategy="test",
            label=name or rule.name,
            lhs=rule.lhs,
            rhs=rule.rhs,
            rule=rule,
        )

    @staticmethod
    def _keys(bank):
        return [alpha_key(r.lhs, r.rhs) for r in bank]

    def test_composite_evaluates_useful(self, bank):
        term, _f = on._case_term(_case("kernel_powscale"))
        rec = on._eval_cand(
            self._cand(POW_MUL3), term, bank, self._keys(bank)
        )
        assert rec["relation"] == "new"
        assert rec["num_true"] is True
        assert rec["derivable"] is True
        assert rec["cand_cost"] < rec["base_cost"]
        assert rec["verdict"] == "useful"

    def test_shipped_law_is_known(self, bank):
        term, _f = on._case_term(_case("kernel_silu"))
        rec = on._eval_cand(
            self._cand(_rule("kern_fuse_silu")),
            term,
            bank,
            self._keys(bank),
        )
        assert rec["relation"] == "duplicate"
        assert rec["verdict"] == "known"

    def test_false_mint_is_false(self, bank):
        # the corpus spells no ``mul(s, s)`` site — hand the mint a
        # term that does (``x·x + y``), so the fp64 oracle can measure
        # the inequality on a real binding.
        from mlopt_core.ir import TensorType, Var

        x = Var("x", TensorType((4, 4)))
        y = Var("y", TensorType((4, 4)))
        term = Op.make("add", Op.make("mul", x, x), y)
        rec = on._eval_cand(
            self._cand(BAD_EQ), term, bank, self._keys(bank)
        )
        assert rec["num_true"] is False
        assert rec["verdict"] == "false"


class TestAdmit:
    """The kernel gauntlet's leg semantics."""

    def test_composite_clears_full(self, ctxs):
        rec = on._admit(POW_MUL3, "full", ctxs, "kernel_powscale")
        assert rec["admitted"]
        assert [s["name"] for s in rec["stages"]] == list(
            on.COMPONENT_ORDER
        )
        assert all(s["passed"] for s in rec["stages"])
        assert rec["measured"]  # the pay leg's per-case ratios

    def test_false_mint_dies_at_oracle(self, ctxs):
        rec = on._admit(BAD_EQ, "full", ctxs, "kernel_strength")
        assert not rec["admitted"]
        assert rec["stages"][0]["name"] == "oracle"
        assert "unequal" in rec["stages"][0]["detail"]

    def test_no_instance_refuses(self, ctxs):
        rec = on._admit(HOPELESS, "full", ctxs, "kernel_powscale")
        assert not rec["admitted"]
        assert rec["stages"][0]["detail"].startswith("no-instance")

    def test_unbound_mint_refuses(self, ctxs):
        """No corpus binding and no sink-lowe​rable instance."""
        rec = on._admit(UNBOUND, "full", ctxs, "kernel_powscale")
        assert not rec["admitted"]
        assert rec["stages"][0]["name"] == "oracle"

    def test_naive_admits_trivially(self, ctxs):
        rec = on._admit(BAD_EQ, "naive", ctxs, "kernel_strength")
        assert rec["admitted"] and rec["stages"] == []

    def test_law_record_is_serializable(self, ctxs):
        rec = on._admit(POW_MUL3, "oracle", ctxs, "kernel_powscale")
        json.dumps(rec)  # the ledger's contract — every cell is JSON


class TestClassify:
    """The sieve verdicts on minted rules."""

    def test_composite_is_known(self, bank, ctxs):
        cres = on._classify(POW_MUL3, bank, ctxs)
        assert cres["verdict"] == "known"
        assert cres["derivable"] is True

    def test_census_mint_unpaid(self, bank, ctxs):
        """``recognize:softmax`` — underivable (its alpha-sibling
        ``softmax_fold`` is excluded as a premise) but the kf_softmax
        floor already owns the case win, so it cannot *beat*."""
        props = {
            p.name: p
            for p in on._corpus_proposals()
            if p.name == "recognize:softmax"
        }
        rule = props["recognize:softmax"].as_rule()
        cres = on._classify(rule, bank, ctxs)
        assert cres["verdict"] in ("unpaid", "known")


class TestRun:
    """The loop end to end on one small episode."""

    def _arm(self):
        def go(state):
            for j in state.pending:
                from kit_discovery.meta_arena import Action

                return Action(
                    "evaluate", {"i": j, "cand": state.pool[j]}
                )
            from kit_discovery.meta_arena import Action

            return Action("stop", {})

        return lambda i: go

    def test_one_episode_funnel(self, tmp_path):
        led = on._Ledger.open(
            tmp_path / "r.jsonl", on._manifest({"t": 1})
        )
        summary = on.run(
            episodes=1,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            rounds=1,
            fuel=64,
            names=["kernel_powscale"],
            cases=[_case("kernel_powscale", page=4, compose=1)],
            ckpt=led,
            log=lambda *a: None,
        )
        assert summary["episodes"] == 1
        assert summary["minted"] >= 1
        assert summary["admitted"] >= 1
        assert summary["leg_refusals"].get("admitted") >= 1
        assert summary["novelty"].get("known", 0) >= 1
        md = on.render_markdown(summary)
        assert "Admission funnel" in md and "Novelty" in md

    def test_resume_replays_cells(self, tmp_path):
        p = tmp_path / "r.jsonl"
        led = on._Ledger.open(p, on._manifest({"t": 1}))
        kw = dict(
            episodes=1,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            rounds=1,
            fuel=64,
            cases=[_case("kernel_powscale", page=4, compose=1)],
            log=lambda *a: None,
        )
        on.run(ckpt=led, **kw)
        n0 = len(led.entries)
        again = on._Ledger.open(p, on._manifest({"t": 1}))
        on.run(ckpt=again, **kw)
        assert len(again.entries) == n0  # nothing recomputed
