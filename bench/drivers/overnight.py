"""The kernel overnight discovery loop — episodes -> mint -> admit -> classify.

The kernel-side port of graphopt's ``bench/drivers/overnight.py``: run
episodes over the registered ``bench.discovery.kernel`` corpus, mint
candidate codegen rewrites from each case's own term, push every mint
through the kernel admission gauntlet (numeric verify -> measured pay
-> adversarial sweep -> certificate — the ``KernelSink`` verify-first
contract), classify the survivors against the known bank through the
novelty sieve's gates, and persist each outcome to an append-only,
protocol-gated JSONL ledger.

Stages:

1. ``episodes`` — one ``_KernelArena`` per case.  The arena is
   ``open_game``-shaped (a pool of minted ``Candidate``s, one
   ``evaluate`` move per pending member, ``stop`` to quit early) and
   the player is an ``open``-board arm (``--player``, default
   ``playbook`` — best heuristic score first, drain).  Each
   ``evaluate`` runs the *kernel* candidate evaluation: bind the
   rule's LHS on the case term, verify the concrete pair through the
   sink (eager bodies — verify-first is about numerics, not codegen
   timing), price the firing in ``kernel_cost``.  Every evaluated
   candidate lands in the ``("episode", i)`` ledger cell.
2. ``mint`` — two sources per case:
   * the **corpus mint**: ``pipeline.propose`` over the shared
     op-tuple census of the whole registered corpus (``"hand"``
     vocabulary — ``"derived"`` needs the parent's ``bench.suites``,
     absent here), bind-gated to this case's term;
   * the **compose mint**: ``meta.synthesize_rules`` — the ``syn_*``
     pair-composition fixpoint — over ``KERNEL_RULES`` plus the
     substrate's *guarded parents* (shipped ``check``/``derive`` laws
     that actually fire on this term), seeded on the case term and
     chained through the emit gate for ``--rounds`` strata.
   Every mint faces the same emit gate: the LHS must strictly bind a
   subterm of the case *and* the instantiated RHS must denote under
   every check-accepted binding (the kernel analogue of the upstream
   ``rhs-err`` drop — the substrate oracle cannot evaluate ``kf_*`` /
   ``om_*`` ops, so the gate is instantiation-only, the numeric work
   deferred to the gauntlet where ``KernelSink`` can see it).
3. ``admit`` — the kernel gauntlet (``--referee`` legs, default
   ``full``): ``oracle`` (the found instance must evaluate ``equal``
   on fp64 feeds through the sink), ``pay`` (firing the rule alone on
   a registered case must extract a *verified* term strictly below
   the spelled cost — the upstream pay leg's shape), ``sweep`` (the
   bound corpus sites all evaluate ``equal`` — no counterexample and
   at least one equal site), ``cert`` (the instance must carry a
   standalone-replayable certificate under the candidate's own rule —
   the sieve's cert semantics).  First refusal ends the run; the
   per-leg verdicts are stored verbatim.
4. ``classify`` — admitted rules get the sieve's three gates via
   ``bench.drivers.screen.novelty_sieve.sieve`` with a minted-candidate
   measurer: the beat arm saturates the case under
   ``KERNEL_RULES + [candidate]`` against the bank-only floor.
   Verdicts are the sieve's own — ``review`` (cert ∧ beat ∧
   ¬derivable — the machine-original class), ``known``, ``refuted``,
   ``uncertified``, ``unpaid``, ``unmeasured``, ``no-instance``.
5. ``ledger`` — an append-only JSONL cell store whose manifest pins
   the protocol; ``--resume`` reopens it (done cells replay, a crash
   costs one cell, a protocol mismatch restarts loudly).
6. ``emit`` — ``docs/results_overnight_kernel.md`` + ``.json``.

The honest question this arm answers: can composed/minted kernel rules
reach fusions or schedules *not* already in ``KERNEL_RULES`` — a fold
the bank lacks — or does composition only ever re-derive what the
bank already proves (the sieve's ``unmeasured``/``unpaid`` classes
mark where the bank is thin).

Run::

    .venv/bin/python -m bench.drivers.overnight --dry
    .venv/bin/python -m bench.drivers.overnight --episodes 12 --resume
    .venv/bin/python -m bench.drivers.overnight \
        --cases kernel_powscale,kernel_chunk_scan --rounds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchkit.checkpoint import Checkpoint, new_manifest
from gamekit import engine
from gamekit.players import registry as players
from kit_discovery import open_game as og
from kit_discovery.verifier import verify_law
from mlopt_core import meta as meta_mod
from mlopt_core.egraph import EGraph, verify_certificate
from mlopt_core.egraph.terms import _term_instantiate, _term_match
from mlopt_core.ir import IR, Op, Var
from mlopt_core.laws import ALL_RULES
from mlopt_core.laws.provenance import law_provenance
from mlopt_core.laws.serialize import (
    alpha_key,
    law_from_data,
    law_to_data,
)
from mlopt_core.meta import (
    _positions,
    _subterm,
    apply_rewrite_at,
    match_pattern,
)

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost

__all__ = ["DEFAULTS", "DRY", "main", "render_markdown", "run"]

#: kernelopt repo root — bench/drivers/overnight.py -> 2 up.
_REPO = Path(__file__).resolve().parents[2]

#: The bounded-run defaults: ten episodes over the registered corpus
#: (registry order), a page-capped pool per episode drained under the
#: same budget, two composition strata.
DEFAULTS: dict[str, Any] = {
    "episodes": 10,
    "page": 24,
    "budget": 24,
    "seed": 0,
    "player": "playbook",
    "referee": "full",
    "rounds": 2,
    "fuel": 256,
}

#: ``--dry`` — seconds-scale smoke: two episodes, a tiny pool, one
#: composition stratum on reduced fuel.
DRY: dict[str, Any] = {
    "episodes": 2,
    "page": 6,
    "budget": 6,
    "rounds": 1,
    "fuel": 64,
}

#: Where the document lands.
DOC_PATH = _REPO / "docs" / "results_overnight_kernel.md"

#: The default ledger — append-only JSONL under the gitignored
#: results tree (``bench/results/`` is scratch, not results).
CKPT_PATH = Path("bench/results/checkpoints/overnight_kernel.jsonl")

#: The census mint's vocabulary — the fixed hand projection.  The
#: ``"derived"`` arm classifies the parent's ``bench.suites`` corpus,
#: which does not exist in this repo.
_MINT_VOCAB = "hand"

#: Per-stratum fuel for the composition mint — each
#: ``synthesize_rules`` round's rule-application budget, the same
#: bound the upstream composition arm was measured under.
_COMPOSE_FUEL = 256

#: The ``meta.family`` a synthesized composite records — distinguishes
#: the pair-composed mints from the census proposers' families.
_COMPOSE_FAMILY = "synth-composite"

#: Cap on bound sites the emit gate walks per minted rule.
_EMIT_SITES = 32

#: Cap on corpus firing sites the sweep leg verifies.
_SWEEP_SITES = 6

#: Cap on registered cases the pay leg probes.
_PAY_CASES = 4

#: Total feed elements above which a truth evaluation declines — the
#: ``kernel_decayed_t2048``/``t4096`` ladder sites would need
#: ~170-400 MB of fp64; over-cap is ``skip``, never a verdict.
_TRUTH_ELEM_CAP = 10_000_000

#: Saturation bounds shared by the pay leg and the minted-measurer —
#: the case terms are small; fixed point arrives in a few iterations.
_SAT_MAX_ITERS = 30
_SAT_MAX_NODES = 10_000

#: The kernel referee arms — strictness dials over the four legs, in
#: gauntlet order (the upstream ``COMPONENT_ORDER`` vocabulary).
COMPONENT_ORDER: tuple[str, ...] = ("oracle", "pay", "sweep", "cert")
REFEREE_LEGS: dict[str, tuple[str, ...]] = {
    "naive": (),
    "oracle": ("oracle",),
    "pay": ("oracle", "pay"),
    "full": COMPONENT_ORDER,
}

_NOVELTY_ORDER = (
    "review",
    "known",
    "unpaid",
    "unmeasured",
    "refuted",
    "uncertified",
    "no-instance",
)


# ---------------------------------------------------------------------------
#  the ledger — benchkit's append-only, protocol-gated JSONL cell store
# ---------------------------------------------------------------------------
#
#  The resume store is ``benchkit.checkpoint.Checkpoint`` (promoted to
#  handl-lang/gamekit ``packages/benchkit``): a manifest-pinned file
#  (protocol must match exactly for entries to replay), atomic
#  O_APPEND record writes, and a torn-tail tolerance so a crash costs
#  one cell, not the run.  ``_manifest`` keeps this lane's provenance
#  spelling — the repo-root ``git`` probe, not the caller's cwd.


def _git(*args: str) -> str:
    """``git`` output in the repo root ("" on failure)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    """A run manifest — protocol pin plus provenance."""
    return new_manifest(
        "overnight-kernel",
        protocol,
        env={
            "git_sha": _git("rev-parse", "--short", "HEAD")
            or "unknown",
            "git_dirty": bool(_git("status", "--porcelain")),
            "device": "cpu",
        },
    )


#: The lane's resume store — ``benchkit.checkpoint.Checkpoint`` under
#: the name the loop's call sites and tests already use.
_Ledger = Checkpoint


# ---------------------------------------------------------------------------
#  the corpus — registered kernel cases as episode boards
# ---------------------------------------------------------------------------


def _registry() -> list[tuple]:
    """``(name, category, term, fires)`` rows — the bench corpus."""
    from bench.discovery.kernel import _cases

    return list(_cases())


def _case_term(case: dict) -> tuple[Any, tuple]:
    """``(term, fires)`` for the case's registered kernel name."""
    for n, _cat, t, fires in _registry():
        if n == case["kernel"]:
            return t, tuple(fires)
    raise KeyError(f"unknown kernel case {case['kernel']!r}")


def gen_cases(
    seed: int,
    episodes: int,
    *,
    page: int,
    rounds: int,
    names: list[str] | None = None,
) -> list[dict]:
    """One case per covered kernel case — registry order, JSON-safe.

    ``names`` selects an explicit subset (``--cases``); otherwise the
    first *episodes* registry rows.  ``rounds`` is the composition
    arm's stratum count — it travels in the case record so a resume's
    pool rebuild reproduces the composites verbatim.
    """
    del seed  # the corpus is fixed; the seed pins feeds, not cases
    rows = _registry()
    if names:
        order = {n: i for i, (n, *_r) in enumerate(rows)}
        picked = [r for r in rows if r[0] in names]
        missing = [n for n in names if n not in order]
        if missing:
            raise KeyError(
                f"unknown kernel case(s) {missing} — "
                f"registered: {[r[0] for r in rows]}"
            )
        rows = picked
    else:
        rows = rows[:episodes]
    return [
        {"kernel": n, "category": cat, "page": page, "compose": rounds}
        for n, cat, _t, _f in rows
    ]


# ---------------------------------------------------------------------------
#  the mint — corpus census proposals + pair-composed kernel rules
# ---------------------------------------------------------------------------


def _op_subterms(term: Any) -> list:
    """Every ``Op`` subterm of *term* — the bind domain."""
    return [s for _p, s in _positions(term) if isinstance(s, Op)]


def _bound_sites(rule: Any, subs: list) -> list:
    """``(subst, subterm)`` pairs where the rule's LHS strictly binds."""
    return [
        (m, s)
        for s in subs
        if (m := _term_match(rule.lhs, s)) is not None
    ][:_EMIT_SITES]


def _site_tag(rule: Any, subst: dict) -> str:
    """The emit gate's per-site tag — the kernel ``rhs-err`` analogue.

    Upstream runs ``evidence._site_outcome`` whose tail evaluates the
    bound pair through the substrate oracle — which cannot lower
    ``kf_*``/``om_*`` ops, so every codegen composite would read
    ``rhs-err`` there.  The kernel gate is instantiation-only: the
    guard preamble is identical (``check`` then ``derive``, exactly
    the e-graph's firing semantics) and the RHS must instantiate —
    numerics are deferred to the gauntlet's ``KernelSink`` oracle.
    """
    try:
        ok = rule.check is None or bool(rule.check(subst))
    except Exception:
        return "guard-err"
    inst = dict(subst)
    if ok and rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            extra = None
        if extra is None:
            ok = False
        else:
            inst.update(extra)
    if not ok:
        return "declined"
    try:
        _term_instantiate(rule.rhs, inst)
    except Exception:
        return "rhs-err"
    return "ok"


def _emittable(rule: Any, subs: list) -> list | None:
    """The bind + denote gate; the accepted sites, or ``None``.

    A mint is kept only when its LHS binds at least one subterm *and*
    no bound site is ``rhs-err`` — an ill-typed composite is dropped
    here, never spending an episode slot or the gauntlet.  Sites the
    guard declines are not failures (a conditional rule's domain is
    its premise).
    """
    sites = _bound_sites(rule, subs)
    if not sites:
        return None
    if any(_site_tag(rule, m) == "rhs-err" for m, _s in sites):
        return None
    return sites


_CORPUS_PROPOSALS: list | None = None


def _corpus_proposals() -> list:
    """The shared census mint — ``pipeline.propose`` over the corpus.

    One op-tuple census over every registered case feeds the proposal
    families (census naturality, mixed views, pattern recognition,
    fusion schemas, shape-aware, grammar).  Minted once per process —
    every episode bind-gates the shared pool against its own term.
    """
    global _CORPUS_PROPOSALS
    if _CORPUS_PROPOSALS is None:
        from kit_discovery import pipeline as pipe
        from kit_discovery.census import CorpusTerm, op_tuple_census

        rows = _registry()
        cts = [CorpusTerm("kernel", n, t) for n, _c, t, _f in rows]
        counts, _terms_of = op_tuple_census(cts)
        _CORPUS_PROPOSALS = list(
            pipe.propose(
                dict(counts), [t for _n, _c, t, _f in rows], _MINT_VOCAB
            )
        )
    return _CORPUS_PROPOSALS


def _guarded_parents(term: Any) -> list:
    """Shipped guarded laws that fire on *term* — the premise parents.

    The composition bank's second half (upstream's ``_guarded_parents``
    verbatim): every ``_synthesizable`` member of ``ALL_RULES``
    carrying ``check``/``derive`` hooks that actually fires somewhere
    on the term.  ``KERNEL_RULES``' own guarded members are already in
    the base bank; these are the substrate singles a guarded
    ``syn_*`` composite descends from.
    """
    from mlopt_core.meta import _synthesizable

    return [
        r
        for r in ALL_RULES
        if (r.check is not None or r.derive is not None)
        and _synthesizable(r)
        and any(
            apply_rewrite_at(r, term, p) is not None
            for p, _s in _positions(term)
        )
    ]


def _mint_composites(
    pool: list,
    meta: dict[int, dict],
    term: Any,
    subs: list,
    case: dict,
    *,
    fuel: int,
) -> None:
    """Pair-compose the kernel bank over the case's own term.

    ``synthesize_rules``'s strata chain through the emit gate — each
    round's gate-surviving composites re-seat as parents for the next
    (enumerated ahead of the base bank so the round's fuel reaches
    them), ending at the fixpoint or *compose* strata.  Pool room is
    whatever the census mint left under ``page``.
    """
    from kit_discovery.proposal import Candidate

    page = case.get("page", DEFAULTS["page"])
    bank = [*KERNEL_RULES, *_guarded_parents(term)]
    seen: set = {
        alpha_key(c.as_rule().lhs, c.as_rule().rhs) for c in pool
    }
    named: set = {c.as_rule().name for c in pool}
    parents: list = []
    for rnd in range(1, int(case.get("compose", 0)) + 1):
        if len(pool) >= page:
            break
        emitted = meta_mod.synthesize_rules(
            [*parents, *bank],
            [term],
            fuel=fuel,
            numeric_check=True,
            max_emitted=page,
        )
        viable: list = []
        for rule in emitted:
            if len(pool) >= page:
                break
            key = alpha_key(rule.lhs, rule.rhs)
            if key in seen:
                continue
            if _emittable(rule, subs) is None:
                continue
            prov = meta_mod.provenance(rule)
            name = f"{rule.name}@{case['kernel']}"
            if name in named:
                name = f"{rule.name}~r{rnd}@{case['kernel']}"
            object.__setattr__(rule, "name", name)
            meta_mod.SYNTH_PARENTS[name] = prov
            pool.append(
                Candidate(
                    strategy="compose",
                    label=name,
                    lhs=rule.lhs,
                    rhs=rule.rhs,
                    note=f"composite: {' then '.join(prov)}",
                    rule=rule,
                )
            )
            meta[len(pool) - 1] = {
                "family": _COMPOSE_FAMILY,
                "sources": list(prov),
                "stratum": rnd,
                "guarded": bool(
                    rule.check is not None or rule.derive is not None
                ),
            }
            seen.add(key)
            named.add(name)
            viable.append(rule)
        if not viable:
            break  # the fixpoint — nothing left to chain on
        parents = viable


class _KernelArena(og.OpenArena):
    """An ``OpenArena`` whose pool is the kernel-minted candidate set.

    The enumerated grammar stream never enters this arm: each episode
    mints its pool *from the case's own term* — the census proposals
    that bind this term plus the ``syn_*`` composites the bank's own
    laws pair-mint over it.  ``meta[i]`` records the proposer family /
    sources / stratum beside the pool, rebuilt deterministically on
    resume.  A mint failure lands in ``self.error``; an empty pool is
    the honest done state.

    ``step`` keeps the open game's move shape (``evaluate`` one
    pending member, ``stop`` to quit) but the evaluation is the
    *kernel* one — a bound firing site verified through the sink and
    priced in ``kernel_cost``, not the substrate oracles the upstream
    arena spends.
    """

    def __init__(
        self, case: dict, *, fuel: int = _COMPOSE_FUEL
    ) -> None:
        from kit_discovery.proposal import Candidate

        self.pool: list = []
        self.meta: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        self.term: Any = None
        self._bank: list = []
        self._lib_keys: list = []
        try:
            self.term, self.fires = _case_term(case)
        except Exception as e:
            self.error = f"case: {type(e).__name__}: {e}"
            return
        subs = _op_subterms(self.term)
        page = case.get("page", DEFAULTS["page"])
        try:
            for p in _corpus_proposals():
                if len(self.pool) >= page:
                    break
                rule = p.as_rule()
                if _emittable(rule, subs) is None:
                    continue
                self.pool.append(
                    Candidate(
                        strategy="corpus",
                        label=p.name,
                        lhs=p.lhs,
                        rhs=p.rhs,
                        note=", ".join(p.sources),
                        rule=rule,
                    )
                )
                self.meta[len(self.pool) - 1] = {
                    "family": getattr(p, "family", ""),
                    "sources": list(getattr(p, "sources", ())),
                }
            if case.get("compose"):
                _mint_composites(
                    self.pool,
                    self.meta,
                    self.term,
                    subs,
                    case,
                    fuel=fuel,
                )
        except Exception as e:
            self.error = f"mint: {type(e).__name__}: {e}"
        self.done = not self.pool

    def _eval_term(self, i: int) -> Any:
        """The term the candidate evaluation binds on — the case's."""
        del i
        return self.term

    def step(self, action: Any) -> tuple[Any, Any]:
        """Evaluate one candidate (``evaluate``) or end (``stop``)."""
        from gamekit.engine import Report

        if self.done:
            raise RuntimeError("episode finished")
        if action.op == "stop":
            self.done = True
            return self.observe(), Report(
                action=action, reward=0.0, terminal=True
            )
        i = action.params["i"]
        rec = _eval_cand(
            self.pool[i],
            self._eval_term(i),
            self._bank,
            self._lib_keys,
        )
        verdict = rec["verdict"]
        self.verdicts[i] = verdict
        self._n_true += verdict in ("true", "useful")
        self._n_useful += verdict == "useful"
        self.done = not any(
            j not in self.verdicts for j in range(len(self.pool))
        )
        self.steps += 1
        reward = og.OPEN_GAME_REWARD["step"] + og.OPEN_GAME_REWARD.get(
            verdict, 0.0
        )
        if verdict == "useful":
            reward += self._efficiency()
        return self.observe(), Report(
            action=action,
            reward=reward,
            terminal=self.done,
            note=f"{verdict}:{self.pool[i].label}",
            detail=rec,
        )


# ---------------------------------------------------------------------------
#  the kernel-prim corpus arm — primitives minted for repeated manual shapes
# ---------------------------------------------------------------------------
#
#  The ``--corpus kernel-prim`` mint: the shared
#  ``kit_discovery.prim_mint.prim_pool`` machinery (promoted out of
#  the parent's ``zoo-prim`` driver — one implementation, both
#  domain repos wire their own hooks) anti-unifies recurring manual
#  subgraphs out of the whole 46-case registry merged into one
#  e-graph, keeps the shapes binding >= ``--xbind`` distinct cases
#  that no kernel-lowerable op already denotes, and mints a NEW
#  ``ksynop_<hash>`` primitive per shape whose denotation IS the
#  spelled subgraph — the binding seats on the sink's own op table
#  (``KernelSink`` lowers it like any fused op).  The emitted
#  candidate is the fold ``pattern -> ksynop_*(args)`` carrying the
#  ``ksynop_*.expand`` definitional premise, so the cert leg's
#  anchored axiom set has what the bare bank lacks and the sieve's
#  derivability gate stays underivable BY CONSTRUCTION — the fold is
#  novel exactly as the upstream ``prim:synop_*`` folds are.


def _prim_knobs(prim: dict | None = None) -> dict:
    """The resolved mining window — defaults overridden by *prim*.

    Same knob vocabulary as the upstream ``zoo-prim`` arm: the
    ``min_ops``/``max_ops``/``cap``/``beam``/``max_expand`` bounds the
    ``--prim-*`` flags write, ``bootstrap`` (``--prim-bootstrap`` —
    seat the core-shipped ``prim:synop_*`` folds into the corpus
    e-graph first, registering their denotations on the kernel table,
    so mined shapes compose over machine-minted primitives),
    ``family_mint``, ``depth`` and ``vocab`` — the semantics live in
    :func:`kit_discovery.prim_mint.prim_pool`.  The resolved dict
    is protocol-pinned (``protocol["prim"]``) and rides the case
    record, so a resume's pool rebuild replays the identical hunt.
    """
    from kit_discovery import prim_mint as _pm

    out = {
        "min_ops": _pm.PRIM_MIN_OPS,
        "max_ops": _pm.PRIM_MAX_OPS,
        "cap": _pm.PRIM_CAP,
        "beam": _pm.PRIM_BEAM,
        "max_expand": _pm.PRIM_MAX_EXPAND,
        "budget": _pm.PRIM_BUDGET,
        "site_cap": _pm.PRIM_SITES,
        "bootstrap": False,
        "family_mint": True,
        "depth": _pm.PRIM_DEPTH,
        "vocab": True,
    }
    if prim:
        out.update({k: v for k, v in prim.items() if v is not None})
    return out


def _denoted_shapes() -> frozenset:
    """Alpha-normal keys of spellings a kernel-lowerable op denotes.

    The kernel *denoted* hook for ``prim_mint.prim_pool``: the known
    bank (kernel bank + kernel prior art + substrate surface) plus
    the raw substrate prior art contribute their bare-call sides —
    restricted to call ops the sink can actually lower
    (``allowed_ops``): a shape named by an op outside the kernel
    table is still a NEW primitive *for this domain*, so only
    kernel-lowerable names count.  A mined shape in the result is
    already named — minting ``ksynop_*`` for it would be a second
    name, so the mint skips it and the funnel records the exclusion.
    """
    from kit_discovery import prim_mint as _pm
    from mlopt_core.laws.priorart import PRIOR_ART_RULES

    from bench.drivers.screen import novelty_sieve as ns
    from mlopt_kernel.sink import kernel_ops

    return _pm.denoted_alphas(
        [*ns.known_bank(), *PRIOR_ART_RULES],
        allowed_ops=kernel_ops(),
    )


def _prim_eval(a: Any, b: Any) -> str:
    """The emit gate's fp64 truth — the ``KernelSink`` oracle's verdict.

    The kernel *eval_pair* hook: ``_truth``'s ``equal`` counts a
    verified site, ``unequal`` refuses the shape, and
    ``indeterminate``/``skip`` are inconclusive (no equal site, no
    measured refute — the gate drops the shape either way).
    """
    verdict, _note = _truth(a, b)
    return verdict if verdict in ("equal", "unequal") else "skip"


def _register_shipped() -> None:
    """Seat the core ``synop*`` denotations on the kernel op table.

    The bootstrap's *register_shipped* hook: each core-shipped
    ``synop*`` op gets its ``expand`` premise's denotation
    registered onto the sink's binding dict, so seated ``synop*``
    enodes lower through ``KernelSink`` exactly like the minted
    ``ksynop_*`` ops do.
    """
    from kit_discovery import prim_mint as _pm
    from mlopt_core.laws import minted

    from mlopt_kernel.sink import _table

    tab = _table()
    for op in minted.SYNOP_SHIPPED:
        fold, expand = minted.prim_fold(op)
        _pm.register_denotation(op, fold.rhs, expand.rhs, tab)


def _prim_pool(
    roots: dict,
    k: int,
    *,
    cap: int,
    budget: int,
    site_cap: int,
    min_ops: int,
    max_ops: int,
    beam: int,
    max_expand: int,
    bootstrap: bool,
    family_mint: bool,
    depth: int,
    vocab: bool,
) -> tuple:
    """The kernel prim-mint pool — the codegen-domain hook wiring.

    :func:`kit_discovery.prim_mint.prim_pool` under kernel hooks:
    the sink's binding table (the ``ksynop_*`` denotations register
    onto it, so every ``KernelSink``/``KernelExecutor`` seated on
    the shared dict lowers them), :func:`_prim_eval` as the emit
    gate's fp64 truth, the kernel bank's denoted set
    (:func:`_denoted_shapes`), the ``ksynop`` prefix, and
    :func:`_register_shipped` as the bootstrap's denotation seater.
    """
    from kit_discovery import prim_mint as _pm

    from mlopt_kernel.sink import _table

    return _pm.prim_pool(
        roots,
        k,
        bindings=_table(),
        eval_pair=_prim_eval,
        denoted=_denoted_shapes(),
        prefix="ksynop",
        handler_bindings=_table(),
        register_shipped=_register_shipped if bootstrap else None,
        cap=cap,
        budget=budget,
        site_cap=site_cap,
        min_ops=min_ops,
        max_ops=max_ops,
        beam=beam,
        max_expand=max_expand,
        bootstrap=bootstrap,
        family_mint=family_mint,
        depth=depth,
        vocab=vocab,
    )


_PPOOL: dict = {}


def _prim_pool_cached(k: int, prim: dict | None = None) -> tuple:
    """The run's shared prim pool — minted once, sliced per episode.

    Minting registers the ``ksynop_*`` ops + sink bindings, so
    calling this before the episode loop is also what makes a
    resume's rebuilt candidate cells evaluable again (names are
    content-hashed — a rebuild re-registers byte-identical
    bindings).  *prim* is the resolved mining window
    (:func:`_prim_knobs`) — part of the cache key.
    """
    knobs = _prim_knobs(prim)
    key = (k, tuple(sorted(knobs.items())))
    if key not in _PPOOL:
        roots = {n: t for n, _c, t, _f in _registry()}
        _PPOOL[key] = _prim_pool(roots, k, **knobs)
    return _PPOOL[key]


def _prim_cases(
    episodes: int, *, page: int, xbind: int, prim: dict | None
) -> list:
    """One case per pool page — the corpus-level mint, paged.

    There is no per-case board on the prim arm: the episode boards
    are ``page``-sized slices of the one shared pool, exactly as the
    upstream ``zoo-prim``/``zoo-xmodule`` arms page theirs.  ``prim``
    (the resolved window) and ``xbind`` ride the case record so a
    resume's arena rebuild replays the identical hunt.
    """
    return [
        {
            "kernel": f"prim-page-{i:02d}",
            "category": "prim",
            "xpage": i,
            "page": page,
            "xbind": xbind,
            "prim": dict(_prim_knobs(prim)),
        }
        for i in range(episodes)
    ]


class _PrimArena(_KernelArena):
    """An ``OpenArena`` whose pool is one page of the prim mint.

    The corpus-level pool (:func:`_prim_pool_cached`) is built once
    per ``(xbind, knobs)`` and episodes slice ``page`` candidates off
    it; ``meta[i]`` rides the ledger with the primitive's own
    provenance (the minted ``ksynop_*`` name, the bound-case set,
    the site-verify counts and the handler verdict).  ``step`` is
    the kernel arena's own — only the evaluated term differs: the
    fold's first bound case's term, read off the mint's module list.
    """

    def __init__(self, case: dict) -> None:
        self.pool: list = []
        self.meta: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        self.term: Any = None
        self.fires: tuple = ()
        self._bank: list = []
        self._lib_keys: list = []
        try:
            cands, meta, _stats = _prim_pool_cached(
                case.get("xbind", 2), case.get("prim")
            )
        except Exception as e:
            self.error = f"mint: {type(e).__name__}: {e}"
            return
        page = case.get("page", DEFAULTS["page"])
        lo = case.get("xpage", 0) * page
        self.pool_total = len(cands)
        self.pool = cands[lo : lo + page]
        self.meta = {
            i: meta[lo + i]
            for i in range(len(self.pool))
            if lo + i in meta
        }
        self._terms = {n: t for n, _c, t, _f in _registry()}
        self.done = not self.pool

    def _eval_term(self, i: int) -> Any:
        """The first bound case's term — where the fold fires."""
        mods = self.meta.get(i, {}).get("prim", {}).get("mods") or []
        if mods:
            return self._terms[mods[0]]
        # no recorded module — fall back to any corpus subterm the
        # fold binds (the emit gate guaranteed at least one).
        for _n, t, *_r in _registry():
            if _first_fire(self.pool[i].as_rule(), t) is not None:
                return t
        return _registry()[0][2]


def _arena_for(case: dict, corpus: str, *, fuel: int = _COMPOSE_FUEL):
    """The episode board for *case* on *corpus*."""
    if corpus == "kernel-prim":
        return _PrimArena(case)
    return _KernelArena(case, fuel=fuel)


def _mint_case(case: dict, meta: dict | None) -> str:
    """The registered case the pay leg anchors on.

    Prim folds mint corpus-wide — the pay leg's guaranteed site is
    the fold's first bound case, read off the mint's module record;
    the census/compose arm's is the episode's own case.
    """
    mods = (meta or {}).get("prim", {}).get("mods") or []
    return mods[0] if mods else case["kernel"]


# ---------------------------------------------------------------------------
#  the kernel candidate evaluation — bind, sink-verify, price
# ---------------------------------------------------------------------------


def _vars_of(t: Any) -> dict:
    """``name -> Var`` over the term's leaves."""
    out: dict = {}
    stack = [t]
    while stack:
        u = stack.pop()
        if isinstance(u, Var):
            out.setdefault(u.name, u)
        elif isinstance(u, Op):
            stack.extend(u.args)
    return out


def _leaf_inputs(*terms: Any) -> tuple[list, str | None]:
    """Collect the leaves of *terms* as one feed list.

    A kernel primitive denotes ONE operation, so a metavar must carry
    one shape across the instance.  An anti-unified corpus mint can
    bind the same name at two shapes (the kernel-prim arm hit 64 vs
    128 at dim 2); feeding that instance unifies the metavar and the
    sink then dies on the broadcast.  Return ``(inputs, note)`` —
    ``note`` names the collision so the caller drops the instance and
    the funnel records it, instead of the run dying on it.
    """
    leaves: dict = {}
    for t in terms:
        for name, var in _vars_of(t).items():
            prev = leaves.get(name)
            if prev is not None:
                a = tuple(prev.typ.shape)
                b = tuple(var.typ.shape)
                if a != b:
                    return [], (
                        f"metavar shape collision: {name} {a} vs {b}"
                    )
            else:
                leaves[name] = var
    return [leaves[n] for n in sorted(leaves)], None


def _truth(lhs: Any, rhs: Any) -> tuple[str, str]:
    """fp64 ``KernelSink`` eval of the concrete pair — eager bodies.

    The kernel truth oracle, sibling to the sieve's
    ``kernel_truth`` but with ``compile_fused=False``: the composite
    bodies run eagerly instead of through ``torch.compile`` — verify
    semantics unchanged (the codegen flag is the *timing* dimension
    the measured bench rows own; the oracle only prices numerics).
    Feeds are fp64 ``randn`` per leaf shape.  Returns ``(verdict,
    note)`` — ``equal`` / ``unequal`` / ``indeterminate`` (non-finite,
    never a refute) / ``skip`` (unlowe​rable op, or over the feed cap).
    """
    import torch

    inputs, note = _leaf_inputs(lhs, rhs)
    if note is not None:
        return "skip", note
    if sum(int(torch.Size(v.typ.shape).numel()) for v in inputs) > (
        _TRUTH_ELEM_CAP
    ):
        return "skip", "over feed cap"
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    names = {v.name for v in inputs}
    sink = KernelSink(compile_fused=False)
    try:
        ref = sink.lower(IR(root=lhs, inputs=inputs, input_names=names))
        opt = sink.lower(IR(root=rhs, inputs=inputs, input_names=names))
        out_r, out_o = ref(*feeds), opt(*feeds)
    except Exception as e:
        return "skip", f"{type(e).__name__}: {e}"
    finite = bool(
        torch.isfinite(out_r).all() and torch.isfinite(out_o).all()
    )
    if not finite:
        return "indeterminate", "non-finite output"
    rep = sink.verify(ref, opt, feeds, rtol=1e-6, atol=1e-8)
    return ("equal" if rep.passed else "unequal"), (
        f"max_abs {rep.max_abs:.2e}"
    )


def _verify_pair(a: Any, b: Any) -> bool | None:
    """``KernelSink.verify`` spelled-vs-extracted on fp64 feeds.

    The pay leg's verify-first step; ``None`` over the feed cap or on
    an unlowe​rable intermediate — inconclusive, never a pass.
    """
    import torch

    inputs, note = _leaf_inputs(a, b)
    if note is not None:
        return None
    if sum(int(torch.Size(v.typ.shape).numel()) for v in inputs) > (
        _TRUTH_ELEM_CAP
    ):
        return None
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    names = {v.name for v in inputs}
    sink = KernelSink(compile_fused=False)
    try:
        rep = sink.verify(
            sink.lower(IR(root=a, inputs=inputs, input_names=names)),
            sink.lower(IR(root=b, inputs=inputs, input_names=names)),
            feeds,
            rtol=1e-4,
        )
    except Exception:
        return None
    return bool(rep.passed)


def _first_fire(rule: Any, term: Any) -> tuple | None:
    """``(path, lhs_i, rhs_i, rewritten)`` — the first firing site."""
    for p, s in _positions(term):
        if match_pattern(rule.lhs, s) is None:
            continue
        out = apply_rewrite_at(rule, term, p)
        if out is not None:
            return p, s, _subterm(out, p), out
    return None


def _relation(lhs: Any, rhs: Any, lib_keys: list) -> str:
    """Alpha-key relation against the bank: duplicate/inverse/new."""
    key = alpha_key(lhs, rhs)
    if key[0] == key[1]:
        return "tautology"
    for rk in lib_keys:
        if key == rk:
            return "duplicate"
        if key == (rk[1], rk[0]):
            return "inverse"
    return "new"


def _eval_cand(
    cand: Any, term: Any, bank: list, lib_keys: list
) -> dict:
    """The kernel candidate evaluation — the arena's ``evaluate``.

    Relation against the known bank first (a duplicate is ``known``
    without spending the oracle); then the bound firing instance is
    sink-verified (``num_true``) and the whole-term ``kernel_cost``
    delta priced.  ``derivable`` is the pattern-level ``verify_law``
    under the known bank — the same triage ``lp.evaluate`` runs
    upstream.  Verdicts are the open game's tiers plus ``unverified``
    (numerics inconclusive and underivable — a ``skip`` is not a
    refute) and ``no-site`` (the emit gate should prevent these; the
    record says so honestly when one slips through).
    """
    from bench.drivers.screen import novelty_sieve as ns

    rule = cand.as_rule()
    rel = _relation(rule.lhs, rule.rhs, lib_keys)
    rec: dict[str, Any] = {
        "label": cand.label,
        "verdict": "",
        "relation": rel,
        "num_true": None,
        "derivable": None,
        "truth_note": "",
        "base_cost": kernel_cost(term),
        "cand_cost": kernel_cost(term),
    }
    if rel != "new":
        rec["verdict"] = "known"
        return rec
    got = _first_fire(rule, term)
    if got is None:
        rec["verdict"] = "no-site"
        return rec
    _path, lhs_i, rhs_i, rewritten = got
    verdict_t, note = _truth(lhs_i, rhs_i)
    rec["num_true"] = {
        "equal": True,
        "unequal": False,
    }.get(verdict_t)
    rec["truth_note"] = f"{verdict_t} {note}".strip()
    rec["cand_cost"] = kernel_cost(rewritten)
    bank = bank if bank else ns.known_bank()
    rec["derivable"] = bool(
        verify_law(rule.lhs, rule.rhs, bank).derivable
    )
    truth = bool(rec["derivable"]) or rec["num_true"] is True
    if not truth:
        rec["verdict"] = (
            "unverified" if rec["num_true"] is None else "false"
        )
    elif rec["cand_cost"] < rec["base_cost"]:
        rec["verdict"] = "useful"
    else:
        rec["verdict"] = "true"
    return rec


def _eval_records(traj: Any) -> list[dict]:
    """One ledger row per ``evaluate`` step of the trajectory."""
    out = []
    for rep in traj.reports:
        if rep.action.op != "evaluate":
            continue
        cand = rep.action.params["cand"]
        out.append(
            {
                "i": rep.action.params["i"],
                "name": cand.as_rule().name,
                "label": rep.detail.get("label", cand.label),
                "verdict": rep.detail.get("verdict", ""),
                "relation": rep.detail.get("relation", ""),
                "num_true": rep.detail.get("num_true"),
                "derivable": rep.detail.get("derivable"),
                "base_cost": rep.detail.get("base_cost"),
                "cand_cost": rep.detail.get("cand_cost"),
            }
        )
    return out


# ---------------------------------------------------------------------------
#  admit — the kernel gauntlet: oracle -> pay -> sweep -> cert
# ---------------------------------------------------------------------------


def _leg_oracle(rule: Any, ctxs: list) -> tuple[dict, tuple | None]:
    """The instance must evaluate ``equal`` through the sink."""
    from bench.drivers.screen import novelty_sieve as ns

    got = ns.instance_of(rule, ctxs)
    if got is None:
        return {
            "name": "oracle",
            "passed": False,
            "detail": "no-instance: gates unevaluable",
        }, None
    src, (lhs_i, rhs_i) = got
    verdict, note = _truth(lhs_i, rhs_i)
    ok = verdict == "equal"
    detail = f"{src}: {verdict}" + (f" ({note})" if note else "")
    if not ok and verdict in ("skip", "indeterminate"):
        detail += " — inconclusive"
    return {"name": "oracle", "passed": ok, "detail": detail}, (
        src,
        lhs_i,
        rhs_i,
    )


def _leg_pay(
    rule: Any, ctxs: list, mint_case: str
) -> tuple[dict, dict]:
    """Fire the rule alone; a verified strict-lower extraction wins.

    The upstream pay leg's kernel shape: the candidate runs *alone*
    over the registered cases it can fire on (the mint case first —
    it is guaranteed a site), the extraction must change the term,
    verify through the sink, and land strictly below the spelled
    ``kernel_cost``.  A measured inequality on a changed extraction
    is a counterexample — a refusal, not just a non-win.
    """
    from bench.drivers.screen import novelty_sieve as ns

    named = [c for c in ctxs if c.registered]
    todo = [c for c in named if c.name == mint_case]
    todo += [
        c
        for c in named
        if c.name != mint_case and ns._relevant(rule, c)
    ]
    rows: list[dict] = []
    for c in todo[:_PAY_CASES]:
        try:
            eg = EGraph()
            root = eg.add_term(c.term)
            eg.run(
                [rule],
                root,
                max_iterations=_SAT_MAX_ITERS,
                max_nodes=_SAT_MAX_NODES,
            )
            fired = sum(
                1 for a in eg.applications if a["rule"] == rule.name
            )
            best = eg.extract_best(root, kernel_cost)
        except Exception as e:
            rows.append({"case": c.name, "error": str(e)})
            continue
        if not fired or best is None or best == c.term:
            continue
        verified = _verify_pair(c.term, best)
        rows.append(
            {
                "case": c.name,
                "fired": fired,
                "cost_before": kernel_cost(c.term),
                "cost_after": kernel_cost(best),
                "ratio": kernel_cost(best) / kernel_cost(c.term),
                "verified": verified,
            }
        )
    wins = [
        r for r in rows if r.get("verified") and r.get("ratio", 9) < 1
    ]
    cex = [r for r in rows if r.get("verified") is False]
    passed = bool(wins) and not cex
    detail = (
        "; ".join(
            f"{r['case']} {r['cost_before']:g}→{r['cost_after']:g} "
            f"(verify {r['verified']})"
            for r in rows
            if "error" not in r
        )
        or "no verified cost-improving extraction"
    )
    if cex:
        detail = "COUNTEREXAMPLE: " + detail
    return {"name": "pay", "passed": passed, "detail": detail}, {
        r["case"]: r["ratio"] for r in rows if "ratio" in r
    }


def _leg_sweep(rule: Any, ctxs: list) -> dict:
    """Every bound corpus site must evaluate equal — the adversarial arm.

    The upstream sweep measures a claim across its accepted region;
    the kernel analogue walks the corpus's firing sites (registered
    cases plus the xcorpus spellings), one sink evaluation each —
    independent fp64 draws per site.  No counterexample and at least
    one equal site, the upstream criterion verbatim.
    """
    from bench.drivers.screen import novelty_sieve as ns

    counts: Counter = Counter()
    seen = 0
    for c in ctxs:
        if seen >= _SWEEP_SITES:
            break
        got = ns._corpus_instance(rule, [c])
        if got is None:
            continue
        _src, (lhs_i, rhs_i) = got
        verdict, _note = _truth(lhs_i, rhs_i)
        counts[verdict] += 1
        seen += 1
    ok = counts["unequal"] == 0 and counts["equal"] >= 1
    detail = (
        f"{counts['equal']}/{seen} equal corpus sites"
        + (
            f" — {counts['unequal']} unequal"
            if counts["unequal"]
            else ""
        )
        + (
            f" ({counts['skip']} skip, {counts['indeterminate']} "
            "indeterminate)"
            if counts["skip"] or counts["indeterminate"]
            else ""
        )
    )
    return {"name": "sweep", "passed": ok, "detail": detail}


def _leg_cert(rule: Any, inst: tuple | None) -> dict:
    """The candidate's own rule must carry a replayable certificate."""
    if inst is None:
        return {
            "name": "cert",
            "passed": False,
            "detail": "no instance to replay",
        }
    _src, lhs_i, rhs_i = inst
    # the anchored axiom set: the candidate's own rule plus its
    # declared premise closure (a ``prim:ksynop_*`` fold's
    # ``ksynop_*.expand`` definitional premise — the upstream cert
    # leg's union, verbatim).
    axioms = [rule, *getattr(rule, "premises", ())]
    res = verify_law(lhs_i, rhs_i, axioms)
    ok = bool(res.derivable and res.replayable)
    detail = (
        f"{res.direction} {res.witness_steps} steps "
        f"({', '.join(res.witness_rules[:4])})"
        if ok
        else res.note or res.stop or "not derivable"
    )
    return {"name": "cert", "passed": ok, "detail": detail}


def _admit(
    rule: Any,
    referee: str,
    ctxs: list,
    mint_case: str,
) -> dict:
    """Run the kernel gauntlet; return the candidate-cell payload.

    Legs run in ``COMPONENT_ORDER`` and stop at the first refusal —
    the gauntlet's own convention.  A composite's record additionally
    carries the ``"guard"`` premise-closure spec
    (``evidence._guard_transport_spec``) when the mint transported
    parent hooks.
    """
    from kit_discovery.evidence import _guard_transport_spec

    law = law_to_data(rule)
    guard = _guard_transport_spec(rule)
    if guard is not None:
        law["guard"] = guard
    premises = getattr(rule, "premises", None)
    if premises:
        # the cert leg's premise-anchored axiom names — the same
        # honesty posture as ``law["guard"]``: record what the
        # closure adds so an audit sees the anchor.  The premise
        # objects themselves re-attach on the deterministic re-mint
        # (``Proposal.premises`` → ``as_rule``).
        law["premises"] = [r.name for r in premises]
    rec: dict[str, Any] = {
        "name": rule.name,
        "provenance": law_provenance(rule),
        "law": law,
        "alpha": [str(k) for k in alpha_key(rule.lhs, rule.rhs)],
        "referee": referee,
        "skipped": "",
        "admitted": False,
        "stages": [],
        "measured": {},
        "reason": "",
    }
    legs = REFEREE_LEGS[referee]
    inst: tuple | None = None
    for leg in legs:
        try:
            if leg == "oracle":
                stage, inst = _leg_oracle(rule, ctxs)
            elif leg == "pay":
                stage, rec["measured"] = _leg_pay(rule, ctxs, mint_case)
            elif leg == "sweep":
                stage = _leg_sweep(rule, ctxs)
            else:
                stage = _leg_cert(rule, inst)
        except Exception as e:  # a component that throws is a refusal
            stage = {
                "name": leg,
                "passed": False,
                "detail": f"{type(e).__name__}: {e}",
            }
        rec["stages"].append(stage)
        if not stage["passed"]:
            rec["reason"] = f"{stage['name']}: {stage['detail']}"
            return rec
    rec["admitted"] = True
    rec["reason"] = "cleared" if legs else "no components ran"
    return rec


# ---------------------------------------------------------------------------
#  classify — the novelty sieve's gates on the admitted mint
# ---------------------------------------------------------------------------


def _measure_minted_case(candidate: Any, ctx: Any) -> Any:
    """``BeatRow``: bank floor vs bank+candidate on one case.

    The sieve's ``_measure_case`` verbatim except the measured arm
    saturates under ``KERNEL_RULES + [candidate]`` — the minted rule
    is not shipped, so the full-bank arm must be told it exists.
    ``beat`` still requires certified ∧ used ∧ strictly below the
    floor ∧ no measured counterexample.
    """
    from bench.drivers.screen import novelty_sieve as ns

    try:
        _fbest, floor, _eg0, _r0 = ns._saturate_cost(
            ctx.term, list(KERNEL_RULES)
        )
    except Exception as e:
        return ns.BeatRow(case=ctx.name, error=f"floor arm: {e}")
    try:
        best, cost_after, eg, _root = ns._saturate_cost(
            ctx.term, [*KERNEL_RULES, candidate]
        )
    except Exception as e:
        return ns.BeatRow(case=ctx.name, error=f"search: {e}")
    fired = sum(
        1 for a in eg.applications if a["rule"] == candidate.name
    )
    try:
        cert = eg.certificate(ctx.term, best)
        verify_certificate(ctx.term, cert)
        certified = True
        used = any(s.rule == candidate.name for s in cert.steps)
    except Exception:
        certified, used = False, False
    verified = ns._verify_extraction(ctx.term, best)
    beat = bool(
        certified
        and used
        and cost_after < floor
        and verified is not False
    )
    return ns.BeatRow(
        case=ctx.name,
        certified=certified,
        verified=verified,
        fired=fired,
        used=used,
        cost_before=kernel_cost(ctx.term),
        cost_after=cost_after,
        floor=floor,
        beat=beat,
    )


def _mint_measurer(
    candidate: Any, ctxs: list, *, max_cases: int = 8
) -> tuple:
    """The measured gate for an unshipped mint — bank+candidate arms.

    Same case selection as the sieve's ``measure_gate`` (registered
    cases the rule is relevant to, capped); the rows are
    :func:`_measure_minted_case`'s.
    """
    from bench.drivers.screen import novelty_sieve as ns

    named = [c for c in ctxs if c.registered]
    todo = [c for c in named if ns._relevant(candidate, c)]
    rows = [
        _measure_minted_case(candidate, c) for c in todo[:max_cases]
    ]
    if not rows or all(r.error for r in rows):
        return None, rows
    return bool(any(r.beat for r in rows)), rows


def _classify(rule: Any, bank: list, ctxs: list) -> dict:
    """The sieve's three gates on a minted rule — its ``to_dict``."""
    from bench.drivers.screen import novelty_sieve as ns

    row = ns.sieve(rule, bank, ctxs=ctxs, measurer=_mint_measurer)
    return row.to_dict()


# ---------------------------------------------------------------------------
#  the loop
# ---------------------------------------------------------------------------


def run(
    *,
    episodes: int,
    page: int,
    budget: int,
    seed: int,
    player: str,
    referee: str,
    rounds: int = 0,
    fuel: int = _COMPOSE_FUEL,
    names: list[str] | None = None,
    ckpt: Any,
    cases: list | None = None,
    arm_factory: Any = None,
    ctxs: list | None = None,
    bank: list | None = None,
    corpus: str = "kernel",
    prim: dict | None = None,
    xbind: int = 2,
    log: Any = print,
) -> dict:
    """Run the loop against *ckpt* (an open ``_Ledger``); summarize.

    ``arm_factory`` is the registry's ``i -> player`` seam — tests
    inject players through it.  ``cases`` overrides the generated
    list (``{"kernel": name, ...}`` dicts).  ``ctxs``/``bank``
    override the sieve's corpus contexts and known bank — tests pass
    reduced slices to keep the run seconds-scale.
    """
    import torch

    from bench.drivers.screen import novelty_sieve as ns

    torch.manual_seed(seed)
    if cases is None:
        if corpus == "kernel-prim":
            cases = _prim_cases(
                episodes, page=page, xbind=xbind, prim=prim
            )
        else:
            cases = gen_cases(
                seed, episodes, page=page, rounds=rounds, names=names
            )
    if ctxs is None:
        ctxs = ns.case_contexts()
    if bank is None:
        bank = ns.known_bank()
    lib_keys = [alpha_key(r.lhs, r.rhs) for r in bank]
    arm = arm_factory or players.make(player, "open", seed=seed, ctx={})
    t0 = time.time()

    for i, case in enumerate(cases):
        ekey = ("episode", i)
        arena = None
        if ckpt.done(ekey):
            evals = ckpt.get(ekey)["evals"]
        else:
            arena = _arena_for(case, corpus, fuel=fuel)
            arena._bank = bank
            arena._lib_keys = lib_keys
            traj = engine.run_episode(arena, arm(i), budget)
            evals = _eval_records(traj)
            payload: dict[str, Any] = {
                "case": case,
                "steps": len(traj.reports),
                "reward": round(traj.total, 6),
                "evals": evals,
            }
            if arena.error:
                payload["error"] = arena.error
            ckpt.record(ekey, payload)
            log(
                f"  episode {i:>3} [{case['kernel']}]: {len(evals)} "
                f"evals reward={traj.total:+.3f}",
            )

        # -- mint + admit: one cell per distinct evaluated rule ----
        pool = None
        for ev in evals:
            if ckpt.done(("candidate", ev["name"])):
                continue
            if pool is None:
                if arena is None:
                    arena = _arena_for(case, corpus, fuel=fuel)
                    arena._bank = bank
                    arena._lib_keys = lib_keys
                pool = arena.pool
            rule = pool[ev["i"]].as_rule()
            rec = _admit(
                rule,
                referee,
                ctxs,
                _mint_case(case, arena.meta.get(ev["i"])),
            )
            meta = arena.meta.get(ev["i"])
            if meta is not None:
                rec["family"] = meta["family"]
                rec["sources"] = meta["sources"]
                if meta.get("stratum"):
                    rec["stratum"] = meta["stratum"]
                rec["guarded"] = bool(meta.get("guarded"))
            rec["label"] = ev["label"]
            rec["episode"] = i
            rec["arena"] = {
                "verdict": ev["verdict"],
                "relation": ev["relation"],
                "num_true": ev["num_true"],
                "derivable": ev["derivable"],
                "base_cost": ev["base_cost"],
                "cand_cost": ev["cand_cost"],
            }
            ckpt.record(("candidate", ev["name"]), rec)
            log(
                f"    candidate {ev['name']:<44} "
                f"admitted={rec['admitted']} ({rec['reason'][:60]})",
            )

        # -- classify: one cell per admitted rule -------------------
        for ev in evals:
            ckey = ("classify", ev["name"])
            if ckpt.done(ckey):
                continue
            crec = ckpt.get(("candidate", ev["name"]))
            if not crec or not crec["admitted"]:
                continue
            rule = law_from_data(crec["law"])
            # The live mint is recovered by index so the sieve judges
            # the rule *with* its hooks — a transported check/derive
            # is code the serialized record cannot carry
            # (``missing_hooks``); the pool rebuilds deterministically.
            if pool is None:
                if arena is None:
                    arena = _arena_for(case, corpus, fuel=fuel)
                    arena._bank = bank
                    arena._lib_keys = lib_keys
                pool = arena.pool
            if (
                ev["i"] < len(pool)
                and pool[ev["i"]].as_rule().name == crec["name"]
            ):
                rule = pool[ev["i"]].as_rule()
            cres = _classify(rule, bank, ctxs)
            cres["name"] = ev["name"]
            ckpt.record(ckey, cres)
            log(
                f"    classify {ev['name']:<42} {cres['verdict']}",
            )

    # -- the ledger summary -----------------------------------------
    names: list[str] = []
    for i in range(len(cases)):
        cell = ckpt.get(("episode", i))
        if not cell:
            continue
        for ev in cell["evals"]:
            if ev["name"] not in names:
                names.append(ev["name"])
    cands = [
        ckpt.get(("candidate", n))
        for n in names
        if ckpt.get(("candidate", n)) is not None
    ]
    leg_deaths: Counter = Counter()
    for rec in cands:
        if rec["skipped"]:
            leg_deaths["skipped"] += 1
            continue
        if rec["admitted"]:
            leg_deaths["admitted"] += 1
            continue
        first = next(
            (s["name"] for s in rec["stages"] if not s["passed"]),
            "error",
        )
        leg_deaths[first] += 1
    novelty: Counter = Counter()
    finds: list[dict] = []
    for rec in cands:
        if not rec["admitted"]:
            continue
        cres = ckpt.get(("classify", rec["name"]))
        if cres is None:
            continue
        novelty[cres["verdict"]] += 1
        if cres["verdict"] == "review":
            finds.append(
                {
                    "name": rec["name"],
                    "instance_src": cres.get("instance_src", ""),
                    "arena_verdict": rec["arena"]["verdict"],
                    "provenance": rec["provenance"],
                }
            )
    arena_v = Counter(
        ev["verdict"]
        for i in range(len(cases))
        for ev in (ckpt.get(("episode", i)) or {}).get("evals", [])
    )
    return {
        "schema": 1,
        "protocol": dict(ckpt.manifest.get("protocol", {})),
        "corpus": "kernel",
        "episodes": len(cases),
        "evaluated": sum(arena_v.values()),
        "arena_verdicts": dict(arena_v),
        "minted": len(cands),
        "composed": sum(
            1 for r in cands if r.get("family") == _COMPOSE_FAMILY
        ),
        "guarded": sum(1 for r in cands if r.get("guarded")),
        "admitted": sum(1 for r in cands if r["admitted"]),
        "leg_refusals": dict(leg_deaths),
        "novelty": dict(novelty),
        "finds": finds,
        "candidates": [
            {
                "name": r["name"],
                "family": r.get("family", ""),
                "episode": r.get("episode"),
                "arena": r["arena"],
                "admitted": r["admitted"],
                "reason": r["reason"],
                "stages": r["stages"],
                "measured": r["measured"],
                "novelty": (
                    ckpt.get(("classify", r["name"])) or {}
                ).get("verdict", ""),
            }
            for r in cands
        ],
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
#  emit
# ---------------------------------------------------------------------------


def _json_path(out: Path) -> Path:
    return (
        out.with_suffix(".json")
        if out.suffix
        else out.parent / f"{out.name}.json"
    )


def render_markdown(summary: dict, *, env: dict | None = None) -> str:
    """The human document — the funnel, the novelty split, the finds."""
    env = env or {}
    legs = summary["leg_refusals"]
    nov = summary["novelty"]
    lines = [
        "# Kernel overnight discovery run",
        "",
        "Episodes -> mint -> admit -> classify on the kernel corpus.  "
        "Each episode's arena pool is minted *from the case's own "
        "term*: the shared corpus-census proposals that bind it plus "
        "the ``syn_*`` pair-composites ``meta.synthesize_rules`` "
        "mints over ``KERNEL_RULES`` (+ the substrate's guarded "
        "parents) seeded on the term.  Every mint faces the emit "
        "gate (strict LHS bind + instantiated RHS must denote).  "
        "Evaluation binds a firing site, verifies the concrete pair "
        "through ``KernelSink`` (eager bodies, fp64) and prices the "
        "firing in ``kernel_cost``.  The admission gauntlet is the "
        "kernel referee — ``oracle`` (instance evaluates ``equal``), "
        "``pay`` (the rule alone must extract a *verified* term "
        "strictly below the spelled cost on a registered case), "
        "``sweep`` (all bound corpus sites equal), ``cert`` (the "
        "instance carries a standalone-replayable certificate).  "
        "Admitted rules are classified by the novelty sieve's gates "
        "— **review** (cert ∧ beat ∧ ¬derivable: the bank cannot "
        "prove it) is the machine-original class.",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')}",
        f"- protocol: `{json.dumps(summary['protocol'], sort_keys=True)}`",
        f"- episodes: {summary['episodes']} · "
        f"evaluated: {summary['evaluated']} · "
        f"minted: {summary['minted']}"
        + (
            f" ({summary['composed']} composites"
            + (
                f", {summary['guarded']} guarded"
                if summary.get("guarded")
                else ""
            )
            + ")"
            if summary.get("composed")
            else ""
        )
        + f" · admitted: {summary['admitted']} · "
        f"wall: {summary['wall_s']}s",
        "- arena verdicts: "
        + (
            ", ".join(
                f"{k}={v}"
                for k, v in sorted(summary["arena_verdicts"].items())
            )
            or "—"
        ),
        "",
        "## Admission funnel",
        "",
        "| first-failed leg | count |",
        "|---|---|",
    ]
    for leg in (*COMPONENT_ORDER, "skipped", "error"):
        if legs.get(leg):
            lines.append(f"| {leg} | {legs[leg]} |")
    lines.append(f"| **admitted** | {legs.get('admitted', 0)} |")
    lines += [
        "",
        "## Novelty",
        "",
        "| verdict | count |",
        "|---|---|",
    ]
    for v in _NOVELTY_ORDER:
        if nov.get(v):
            lines.append(f"| {v} | {nov.get(v, 0)} |")
    lines += ["", "## Machine-original finds (sieve `review`)", ""]
    if summary["finds"]:
        for f in summary["finds"]:
            lines.append(
                f"- `{f['name']}` — arena `{f['arena_verdict']}`, "
                f"instance {f['instance_src'] or '—'}, "
                f"provenance {f['provenance'] or '—'}"
            )
    else:
        lines.append(
            "**None.** Every admitted mint was already a theorem of "
            "the known bank — or nothing survived the gauntlet.  The "
            "funnel above is the measurement: composed kernel rules "
            "are derivable by construction when both parents are in "
            "the bank, so the honest expectation is refusals at "
            "``pay`` (a composite never beats the floor its own "
            "chain reaches) and ``known`` verdicts at classify."
        )
    lines += [
        "",
        "## Candidates",
        "",
        "| candidate | family | arena | admitted | reason | novelty |",
        "|---|---|---|---|---|---|",
    ]
    for c in summary["candidates"]:
        reason = (c["reason"] or "—").replace("|", "\\|")
        lines.append(
            f"| `{c['name']}` | {c.get('family') or '—'} "
            f"| {c['arena'].get('verdict', '—')} "
            f"| {'yes' if c['admitted'] else 'no'} "
            f"| {reason} | {c['novelty'] or '—'} |"
        )
    return "\n".join(lines) + "\n"


def _env() -> dict:
    """The provenance block — timestamp, git sha, torch, device."""
    import datetime as _dt

    try:
        import torch

        tver = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        tver, device = "?", "?"
    return {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "torch": tver,
        "device": device,
    }


def emit(summary: dict, out: Path, *, env: dict) -> Path:
    """Write ``out`` (.md) and the sibling .json."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summary, env=env))
    payload = {"env": env, **summary}
    _json_path(out).write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return out


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the kernel overnight loop; emit the result documents."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        type=int,
        default=DEFAULTS["episodes"],
        help="episode count — the first N registered kernel cases",
    )
    ap.add_argument(
        "--page",
        type=int,
        default=DEFAULTS["page"],
        help="per-episode pool cap (minted candidates)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=DEFAULTS["budget"],
        help="per-episode action budget — >= page drains the pool",
    )
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument(
        "--player",
        default=DEFAULTS["player"],
        help="a registered players/ name on the open board "
        "(default playbook — best heuristic score first, drain)",
    )
    ap.add_argument(
        "--referee",
        choices=sorted(REFEREE_LEGS),
        default=DEFAULTS["referee"],
        help="the kernel gauntlet arm — legs run in gauntlet order "
        f"{COMPONENT_ORDER} (default full)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=DEFAULTS["rounds"],
        metavar="ROUNDS",
        help="composition strata — each round's gated composites "
        "re-seat as parents until the fixpoint (0 disables the "
        "compose arm; the census mint still runs)",
    )
    ap.add_argument(
        "--fuel",
        type=int,
        default=DEFAULTS["fuel"],
        help="per-stratum rule-application budget for the mint",
    )
    ap.add_argument(
        "--cases",
        default=None,
        help="comma-separated bench.discovery.kernel case names — "
        "overrides --episodes (registry order kept)",
    )
    ap.add_argument(
        "--dry",
        action="store_true",
        help="seconds-scale smoke: 2 episodes, page 6, one stratum",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=f"append-only ledger (default {CKPT_PATH})",
    )
    ap.add_argument(
        "--corpus",
        choices=["kernel", "kernel-prim"],
        default="kernel",
        help="kernel: one episode per registered case; "
        "kernel-prim: the prim mint — prim_mint anti-unifies the "
        "merged registry, mints ksynop_* primitives per recurring "
        "manual shape, and episodes page the shared pool",
    )
    ap.add_argument(
        "--prim-min-ops",
        type=int,
        default=None,
        help="kernel-prim only: mining window floor",
    )
    ap.add_argument(
        "--prim-max-ops",
        type=int,
        default=None,
        help="kernel-prim only: mining window cap",
    )
    ap.add_argument(
        "--prim-cap",
        type=int,
        default=None,
        help="kernel-prim only: pool size cap",
    )
    ap.add_argument(
        "--xbind",
        type=int,
        default=2,
        help="kernel-prim only: minimum distinct cases a minted "
        "primitive must bind (the cross-case surface)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse the ledger's done cells (protocol-verified); "
        "without it the ledger restarts fresh",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"markdown output (default {DOC_PATH}); "
        "the .json lands beside it",
    )
    args = ap.parse_args(argv)
    if args.rounds < 0:
        ap.error("--rounds must be >= 0")
    if args.corpus != "kernel-prim" and (
        args.prim_min_ops is not None
        or args.prim_max_ops is not None
        or args.prim_cap is not None
    ):
        ap.error("--prim-* knobs require --corpus kernel-prim")
    prim = {
        k: v
        for k, v in {
            "min_ops": args.prim_min_ops,
            "max_ops": args.prim_max_ops,
            "cap": args.prim_cap,
        }.items()
        if v is not None
    }
    if args.dry:
        for k, v in DRY.items():
            setattr(args, k, v)
    names = (
        [n.strip() for n in args.cases.split(",") if n.strip()]
        if args.cases
        else None
    )
    if args.ckpt is None:
        args.ckpt = CKPT_PATH
    if args.out is None:
        args.out = DOC_PATH

    #  ``python bench/drivers/overnight.py`` (non-``-m``) needs the
    #  repo root on sys.path before the function-level ``bench.*``
    #  imports resolve.
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    protocol = {
        "episodes": args.episodes,
        "page": args.page,
        "budget": args.budget,
        "seed": args.seed,
        "player": args.player,
        "referee": args.referee,
        "rounds": args.rounds,
        "fuel": args.fuel,
        "cases": names,
        # the mint semantics — a ledger minted under a different
        # emit gate or vocabulary must restart loudly.
        "mint": "kernel-corpus-compose",
        "vocab": _MINT_VOCAB,
        "corpus": args.corpus,
        "prim": _prim_knobs(prim)
        if args.corpus == "kernel-prim"
        else None,
        "dry": bool(args.dry),
    }
    if not args.resume and args.ckpt.exists():
        args.ckpt.unlink()
    ckpt = _Ledger.open(args.ckpt, _manifest(protocol))
    if ckpt.entries:
        print(  # stdout-compat
            f"resume: {len(ckpt.entries)} cells already done "
            f"({args.ckpt})",
            flush=True,
        )

    print(  # stdout-compat
        f"== overnight-kernel — {args.episodes} episodes, "
        f"page {args.page}, player {args.player}, "
        f"referee {args.referee}, rounds {args.rounds} =="
    )
    summary = run(
        episodes=args.episodes,
        page=args.page,
        budget=args.budget,
        seed=args.seed,
        player=args.player,
        referee=args.referee,
        rounds=args.rounds,
        fuel=args.fuel,
        names=names,
        ckpt=ckpt,
        corpus=args.corpus,
        prim=prim or None,
        xbind=args.xbind,
    )

    env = _env()
    out = emit(summary, args.out, env=env)
    print(f"wrote {out} (+ {_json_path(out).name})")  # stdout-compat
    print(  # stdout-compat
        f"episodes={summary['episodes']} "
        f"evaluated={summary['evaluated']} minted={summary['minted']} "
        f"admitted={summary['admitted']} "
        f"novelty={summary['novelty']} wall={summary['wall_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
