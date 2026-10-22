"""The kernel-side novelty sieve — cert ∧ beat ∧ ¬derivable.

The codegen port of graphopt's
``bench/drivers/screen/novelty_sieve.py``: a ``KERNEL_RULES`` law is a
**novelty candidate** iff it passes three gates —

1. **cert** — the law certifies itself: on a real firing instance its
   own rewrite derives the instance sides with a
   standalone-replayable certificate (the shared substrate's
   ``mlopt_discovery.verifier.verify_law`` + ``verify_certificate`` —
   reused, not rebuilt).
2. **beat** — the law *pays*: on at least one registered
   ``bench.discovery.kernel`` case, saturating the spelled term under
   the full kernel bank extracts a certified term whose
   :func:`~mlopt_kernel.cost.kernel_cost` lands **strictly below the
   floor** — the best extraction the *rest* of the kernel bank (the
   candidate and its alpha-siblings removed) achieves on the same
   term, itself never worse than the spelled cost — AND the
   derivation certificate invokes the candidate (a floor the law
   never touched is not the law's win).
3. **derivable** — the law is NOT already a theorem of the known
   bank: the instance saturated under the bank with the candidate —
   and every alpha-sibling (same ``(lhs, rhs)`` under metavariable
   renaming, keyed by ``laws.serialize.alpha_key``) — removed.

The known bank is the kernel bank plus the **shared substrate
surface**: ``default_rules()`` + ``ALL_RULES_WITH_LAYOUT`` +
``ATTENTION_RULES`` + the scan monoids + the prior-art
``FUSION``/``CATEGORICAL`` slice — the same widest floor the upstream
sieve saturates against, minus the ``EXPANSIVE`` rules that explode
its bounded e-graph.  The substrate premises belong because the two
banks rewrite the *same* IR: a codegen law the tensor bank already
proves is genuinely known — ``kern_square_strength`` self-declares
``("pow_to_square", "square_expand")`` and demotes to ``known``
against this bank, exactly as it should.  What the substrate bank
cannot do is mint a ``kf_*``/``om_*`` op — no tensor rule emits the
kernel vocabulary — so the fused-op and scheduled-scan folds are
underivable by construction here; the sieve *measures* that claim
rather than asserting it.  The discovery-side minted pool
(``lawdata.CANDIDATE_LAWS`` / ``impact.new_laws``) is deliberately
absent: kernelopt ships no minted candidates, and the parent repo's
unshipped machinery is not this bank's premise.

Verdicts: ``review`` (all three gates — enqueued for a human),
``known`` (bank-derivable), ``refuted`` (the fp64 ``KernelSink``
oracle measured the instance unequal), ``uncertified``, ``unpaid``
(underivable but no case win), ``unmeasured`` (underivable but no
registered case exercises it), ``no-instance`` (no binding produced a
site — untestable, reported honestly).

Instance sources, in order: the law_bench registry
(``verifier.instance_of`` — absent in this repo since the registry
lives in the parent's ``bench.suites``; guarded, not assumed), a
strict LHS match inside the kernel corpus — the registered
``bench.discovery.kernel`` cases plus a small *extended* corpus of
flat n-ary chunk-scan spellings (``xcorpus``, instance-finding only —
the registered bench spells concat as a nested binary tree, so the
3-6-ary family members have no registered site) — the gauntlet's
synthesized binding domain (``evidence._synth_sites``), and finally a
uniform-shape metavariable binding.

The truth oracle is ``KernelSink.verify`` on fp64 feeds — the shared
``mlopt_discovery.oracle`` has no ``kf_*``/``om_*`` bindings.  A
non-finite evaluation is ``indeterminate``, never a refute.

Run::

    .venv/bin/python -m bench.drivers.screen.novelty_sieve
    .venv/bin/python -m bench.drivers.screen.novelty_sieve \
        --names kern_fuse_chunk_scan,kern_square_strength --no-measure
    .venv/bin/python -m bench.drivers.screen.novelty_sieve \
        --json /tmp/sieve.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlopt_core.egraph import EGraph, verify_certificate
from mlopt_core.ir import IR, Const, Op, TensorType, Var, op_repr
from mlopt_core.laws import (
    ALL_RULES_WITH_LAYOUT,
    ATTENTION_RULES,
    PRIOR_ART,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    tags,
)
from mlopt_core.laws.provenance import law_provenance
from mlopt_core.laws.serialize import alpha_key
from mlopt_core.meta import (
    _positions,
    _subterm,
    apply_rewrite_at,
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from mlopt_discovery import lawdata, verifier
from mlopt_discovery.verifier import verify_law
from mlopt_orchestrator.optimize import default_rules

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost

__all__ = [
    "BeatRow",
    "SieveRow",
    "case_contexts",
    "instance_of",
    "kernel_truth",
    "known_bank",
    "main",
    "measure_gate",
    "render_markdown",
    "shipped_candidates",
    "sieve",
    "sieve_all",
]

#: kernelopt repo root — bench/drivers/screen/novelty_sieve.py → 3 up.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Where the review-queue document lands.
DOC_PATH = REPO_ROOT / "docs" / "results_kernel_sieve.md"

#: Cap on the synthesized-binding enumeration per law
#: (``evidence._synth_sites`` window — same order as the gauntlet's).
_SYNTH_LIMIT = 400

#: Per-candidate cap on measured cases.
_MAX_CASES = 8

#: Uniform-shape fallbacks for patterns no corpus site binds.
_LOOSE_SHAPES = ((4, 4), (2, 4), (4,), (1, 4), (4, 1), (2, 2, 4))
_LOOSE_ATTR_DOM = (-1, -2, 0, 1, 2)

#: Saturation bounds for the measured gate's two e-graph arms — the
#: case terms are small; fixed point arrives in a few iterations.
_MEASURE_MAX_ITERS = 30
_MEASURE_MAX_NODES = 10_000

#: Total feed elements above which the measured gate skips its
#: informational ``KernelSink.verify`` — the ``kernel_decayed_t2048``/
#: ``t4096`` ladder rows would need ~170-400 MB of fp64 (and a
#: ~70-137 GFLOP fp64 matmul) for a number the timed row already
#: owns.  Every other registered case verifies.
_VERIFY_ELEM_CAP = 10_000_000


# ---------------------------------------------------------------------------
#  The known bank + the candidate pool
# ---------------------------------------------------------------------------


def _substrate_bank() -> list:
    """The shared substrate surface — the widest shipped tensor bank.

    Same composition as the upstream sieve's ``known_bank`` slice:
    the default bank plus layout/attention/scan rules plus the
    prior-art ``FUSION``/``CATEGORICAL`` rules (the structural
    transforms that are the novelty-floor content).  ``EXPANSIVE``
    prior-art rules are excluded — they cycle on neg/add/sub/mul and
    blow the bounded e-graph's node budget (upstream measured the
    composite demotion exceeding a 200 s box without the cut).
    """
    seen: dict[str, Any] = {}
    for group in (
        default_rules(),
        ALL_RULES_WITH_LAYOUT,
        ATTENTION_RULES,
        SCAN_LAWS,
        SCAN_DIAG_LAWS,
    ):
        for r in group:
            seen.setdefault(r.name, r)
    for r in PRIOR_ART:
        if tags.EXPANSIVE in r.tags:
            continue
        if tags.FUSION in r.tags or tags.CATEGORICAL in r.tags:
            seen.setdefault(r.name, r)
    return list(seen.values())


def known_bank() -> list:
    """The derivability floor: ``KERNEL_RULES`` U the substrate bank.

    Kernel names win on collision (none exist today — every
    ``kern_*`` name is distinct from the substrate's), deduplicated
    by name.
    """
    seen: dict[str, Any] = {}
    for r in itertools.chain(KERNEL_RULES, _substrate_bank()):
        seen.setdefault(r.name, r)
    return list(seen.values())


def shipped_candidates() -> list:
    """The sweep population — the whole shipped ``KERNEL_RULES`` bank.

    Unlike the upstream sieve there is no minted-but-unshipped pool
    here: every codegen law in the bank is a candidate.
    """
    return list(KERNEL_RULES)


def _family(name: str) -> str:
    """The doc's per-family grouping, by name convention."""
    if name.startswith("kern_fuse_chunk_scan"):
        return "schedule"
    if name.startswith("kern_fuse_"):
        return "fusion"
    if name.startswith("kern_"):
        return "strength"
    return "other"


# ---------------------------------------------------------------------------
#  Instances — where the law's two sides come from
# ---------------------------------------------------------------------------


def _vars_of(t: Any) -> list:
    """DFS the term's ``Var`` leaves (dedup by name)."""
    seen: dict[str, Var] = {}
    stack = [t]
    while stack:
        u = stack.pop()
        if isinstance(u, Var):
            seen.setdefault(u.name, u)
        elif isinstance(u, Op):
            stack.extend(u.args)
    return list(seen.values())


@dataclass(frozen=True)
class _CaseCtx:
    """A corpus term: a registered kernel case or an xcorpus site."""

    name: str
    term: Any
    fires: tuple = ()  # the case's scripted law names (signal)
    registered: bool = True  # only registered cases feed the beat gate


def _scan_arm(i: int, *, c: int = 8, dv: int = 6, b: int = 4) -> Any:
    """One flat chunk arm — the shape ``is_chunk_scan`` parses.

    Mirrors ``tests/test_kernel.py``'s ``_scan_arm``: an intra
    ``om_decayed`` element under a scale peel, plus one decay-gated
    pair product ``row ⊙ (q_i @ (a_j @ v_j)) / S`` per earlier block.
    """
    g = Var("g", TensorType(()))
    intra = Op.make(
        "om_decayed",
        Var(f"s{i}", TensorType((b, c, c))),
        Var(f"d{i}", TensorType((c, c))),
        Var(f"v{i}", TensorType((b, c, dv))),
    )
    terms = [Op.make("div", intra, Const(0.5))]
    for j in range(i):
        row = Op.make("unsqueeze", Op.make("decay_vec", g, t=c), dim=-1)
        terms.append(
            Op.make(
                "mul",
                row,
                Op.make(
                    "div",
                    Op.make(
                        "matmul",
                        Var(f"q{i}", TensorType((b, c, c))),
                        Op.make(
                            "matmul",
                            Var(f"a{j}", TensorType((b, c, c))),
                            Var(f"v{j}", TensorType((b, c, dv))),
                        ),
                    ),
                    Const(0.5),
                ),
            )
        )
    arm = terms[0]
    for t in terms[1:]:
        arm = Op.make("add", arm, t)
    return arm


def _xcorpus() -> list[_CaseCtx]:
    """Flat n-ary chunk-scan spellings — instance sources only.

    The registered bench's one scan case spells the concat nested
    binary, so ``kern_fuse_chunk_scan_{3..6}`` have no registered
    site; these contexts give the cert/derivability gates a real
    firing instance without pretending the n-ary spellings are
    registered measurements.
    """
    return [
        _CaseCtx(
            name=f"xcorpus:scan_{n}",
            term=Op.make(
                "concat", *[_scan_arm(i) for i in range(n)], dim=-2
            ),
            registered=False,
        )
        for n in (3, 4, 5, 6)
    ]


def case_contexts() -> list[_CaseCtx]:
    """The kernel corpus: registered cases + the xcorpus sites.

    Case terms are spelled IR already — no fixture build, no torch
    export — so contexts are cheap and *always* available to the
    instance search (unlike upstream, where building a context costs
    a torch trace and is gated behind ``measure``).
    """
    from bench.discovery.kernel import _cases

    ctxs = [
        _CaseCtx(name=n, term=t, fires=tuple(fires))
        for n, _cat, t, fires in _cases()
    ]
    return ctxs + _xcorpus()


def _corpus_instance(
    rule: Any, ctxs: list[_CaseCtx]
) -> tuple[str, tuple[Any, Any]] | None:
    """First strict LHS match inside the kernel corpus."""
    for ctx in ctxs:
        for path, sub in _positions(ctx.term):
            if match_pattern(rule.lhs, sub) is None:
                continue
            out = apply_rewrite_at(rule, ctx.term, path)
            if out is not None:
                return ctx.name, (sub, _subterm(out, path))
    return None


def _synth_instance(rule: Any, limit: int) -> tuple[Any, Any] | None:
    """A guard-accepted site from the gauntlet's binding enumeration."""
    from mlopt_discovery.evidence import _synth_sites

    for bound, lhs_t in _synth_sites(rule.lhs, rule.rhs, limit=limit):
        try:
            if rule.check is not None and not rule.check(bound):
                continue
        except Exception:
            continue
        out = apply_rewrite_at(rule, lhs_t, ())
        if out is not None:
            return lhs_t, out
    return None


def _loose_instance(rule: Any) -> tuple[Any, Any] | None:
    """Uniform-shape metavariable binding — the last-resort instance.

    Same posture as upstream: leaf metavars get ``Var``s of a trial
    shape (scalar-metavariable names mint ``Const``s per the lawdata
    table), ``$attr:`` names sweep a small dim domain, and the rule's
    own ``check``/``derive`` vetoes bindings that cannot denote.
    """
    mvs = sorted(pattern_metavars(rule.lhs))
    attr_mvs = [m for m in mvs if m.startswith("$attr:")]
    leaf_mvs = [m for m in mvs if not m.startswith("$attr:")]
    attr_doms = [
        [lawdata.INSTANCE_ATTR_DEFAULTS[m[len("$attr:") :]]]
        if m[len("$attr:") :] in lawdata.INSTANCE_ATTR_DEFAULTS
        else list(_LOOSE_ATTR_DOM)
        for m in attr_mvs
    ]
    combos = itertools.product(*attr_doms) if attr_doms else [()]
    for combo in combos:
        for shape in _LOOSE_SHAPES:
            subst: dict[str, Any] = dict(
                zip(attr_mvs, combo, strict=True)
            )
            for mv in leaf_mvs:
                if mv in lawdata.INSTANCE_SCALAR_MVARS:
                    subst[mv] = Const(lawdata.INSTANCE_SCALAR_MVARS[mv])
                else:
                    subst[mv] = Var(mv, TensorType(shape))
            try:
                if rule.check is not None and not rule.check(subst):
                    continue
                lhs = instantiate_pattern(rule.lhs, subst)
                inst = dict(subst)
                if rule.derive is not None:
                    extra = rule.derive(subst)
                    if extra is None:
                        continue
                    inst.update(extra)
                return lhs, instantiate_pattern(rule.rhs, inst)
            except Exception:
                continue
    return None


def instance_of(
    rule: Any,
    ctxs: list[_CaseCtx],
    *,
    synth_limit: int = _SYNTH_LIMIT,
) -> tuple[str, tuple[Any, Any]] | None:
    """Return ``(source, (lhs, rhs))`` — a real firing site of *rule*.

    Sources in order: ``registry`` (the shared
    ``verifier.instance_of`` — its law_bench table lives in the
    parent's ``bench.suites`` and is absent in this repo, so the call
    is guarded and simply never fires here), ``corpus:<case>`` /
    ``xcorpus:<site>`` (a strict match in the kernel corpus),
    ``synth`` (the gauntlet's binding enumeration), ``loose``
    (uniform-shape binding).  ``None`` when no source produces a
    site.
    """
    try:
        inst = verifier.instance_of(rule)
    except Exception:
        inst = None
    if inst is not None:
        return "registry", inst
    got = _corpus_instance(rule, ctxs)
    if got is not None:
        cname, inst = got
        return f"corpus:{cname}", inst
    inst = _synth_instance(rule, synth_limit)
    if inst is not None:
        return "synth", inst
    inst = _loose_instance(rule)
    if inst is not None:
        return "loose", inst
    return None


# ---------------------------------------------------------------------------
#  The truth oracle — KernelSink.verify on fp64 feeds
# ---------------------------------------------------------------------------


def kernel_truth(lhs: Any, rhs: Any) -> tuple[str, str]:
    """Numerically evaluate *lhs* vs *rhs* through ``KernelSink``.

    The substrate ``oracle.eval_instance`` has no ``kf_*``/``om_*``
    bindings; the kernel sink is the domain's executable semantics —
    the same verify-first surface the delivery arm uses.  Feeds are
    fp64 ``randn`` per leaf shape.  Returns ``(verdict, note)`` with
    verdict in ``equal`` / ``unequal`` / ``indeterminate`` (a
    non-finite evaluation — inconclusive, never a refute) / ``skip``
    (a term the sink cannot lower — e.g. a loose instance binding an
    op outside the kernel table).
    """
    import torch

    sink = KernelSink()
    leaves = {v.name: v for v in _vars_of(lhs)}
    leaves.update({v.name: v for v in _vars_of(rhs)})
    inputs = [leaves[n] for n in sorted(leaves)]
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    names = {v.name for v in inputs}
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
    verdict = "equal" if rep.passed else "unequal"
    return verdict, f"max_abs {rep.max_abs:.2e}"


# ---------------------------------------------------------------------------
#  The derivability gate — is the law already a theorem of the bank?
# ---------------------------------------------------------------------------


def _premises(
    candidate: Any, bank: list
) -> tuple[list, tuple[str, ...]]:
    """The premise bank: *bank* minus the candidate's alpha-siblings.

    Identical semantics to upstream: a sibling shares the
    candidate's ``alpha_key`` (the same law under metavariable
    renaming — leaving one in would self-derive); the inverse
    spelling is a different law and stays.
    """
    key = alpha_key(candidate.lhs, candidate.rhs)
    sibs = tuple(r.name for r in bank if alpha_key(r.lhs, r.rhs) == key)
    premises = [r for r in bank if alpha_key(r.lhs, r.rhs) != key]
    return premises, sibs


# ---------------------------------------------------------------------------
#  The measured gate — beat the rest-of-bank floor, with attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatRow:
    """One registered case's measured-gate outcome for a candidate."""

    case: str
    certified: bool = False
    verified: bool | None = None  # sink.verify spelled-vs-extracted
    fired: int = 0  # e-graph applications of the candidate
    used: bool = False  # candidate appears in the cert's steps
    cost_before: float | None = None  # spelled kernel_cost
    cost_after: float | None = None  # extraction under the full bank
    floor: float | None = None  # extraction under bank - candidate
    beat: bool = False  # certified ∧ used ∧ cost_after < floor
    error: str = ""


def _ops_of(term: Any) -> frozenset:
    """Op names reachable in *term*."""
    out: set[str] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            out.add(t.op)
            stack.extend(t.args)
    return frozenset(out)


def _relevant(candidate: Any, ctx: _CaseCtx) -> bool:
    """Whether *ctx* can exercise *candidate*.

    A scripted-fire signal (the case's ``rules_to_fire`` names the
    candidate — the kernel analogue of upstream's ``law:<name>``
    signal); otherwise the candidate's LHS must strictly match some
    subterm of the spelled term.
    """
    if candidate.name in ctx.fires:
        return True
    for _path, sub in _positions(ctx.term):
        if match_pattern(candidate.lhs, sub) is not None:
            return True
    return False


def _saturate_cost(
    term: Any, rules: list
) -> tuple[Any, float, EGraph, int]:
    """Saturate *term* under *rules*; return (best, cost, eg, root)."""
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        rules,
        root,
        max_iterations=_MEASURE_MAX_ITERS,
        max_nodes=_MEASURE_MAX_NODES,
    )
    best = eg.extract_best(root, kernel_cost) or term
    return best, kernel_cost(best), eg, root


def _verify_extraction(term: Any, best: Any) -> bool | None:
    """``KernelSink.verify`` spelled-vs-extracted on fp64 feeds.

    ``None`` when the feed would exceed ``_VERIFY_ELEM_CAP`` — the
    number is informational (the timed bench row owns the measured
    arm); the beat gate runs on the certificate alone, same as
    upstream.
    """
    import torch

    leaves = {v.name: v for v in _vars_of(term)}
    leaves.update({v.name: v for v in _vars_of(best)})
    inputs = [leaves[n] for n in sorted(leaves)]
    if sum(int(torch.Size(v.typ.shape).numel()) for v in inputs) > (
        _VERIFY_ELEM_CAP
    ):
        return None
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    names = {v.name for v in inputs}
    sink = KernelSink()
    try:
        rep = sink.verify(
            sink.lower(IR(root=term, inputs=inputs, input_names=names)),
            sink.lower(IR(root=best, inputs=inputs, input_names=names)),
            feeds,
            rtol=1e-4,
        )
    except Exception:
        return None
    return bool(rep.passed)


def _measure_case(candidate: Any, ctx: _CaseCtx) -> BeatRow:
    """Saturate the case term twice: bank-minus-candidate vs bank.

    The floor arm is the honest codegen baseline: the best
    ``kernel_cost`` extraction the rest of ``KERNEL_RULES`` reaches
    on the same spelled term — always ≤ the spelled cost, and equal
    to it when no peer law helps.  ``beat`` requires the certified
    extraction under the full bank to land strictly below that floor
    *and* the derivation certificate to name the candidate — a win
    the rule never touched is not the rule's win.
    """
    premises, _sibs = _premises(candidate, list(KERNEL_RULES))
    try:
        _fbest, floor, _eg0, _r0 = _saturate_cost(ctx.term, premises)
    except Exception as e:
        return BeatRow(case=ctx.name, error=f"floor arm: {e}")
    try:
        best, cost_after, eg, _root = _saturate_cost(
            ctx.term, list(KERNEL_RULES)
        )
    except Exception as e:
        return BeatRow(case=ctx.name, error=f"search: {e}")
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
    verified = _verify_extraction(ctx.term, best)
    # ``verified`` is informational: a measured *failure* (False) is a
    # counterexample and vetoes the win; ``None`` (over the feed cap,
    # or an unlowe​rable intermediate) is inconclusive — the
    # certificate carries it, same as upstream where the beat gate is
    # cert-replay only.
    beat = bool(
        certified
        and used
        and cost_after < floor
        and verified is not False
    )
    return BeatRow(
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


def measure_gate(
    candidate: Any,
    ctxs: list[_CaseCtx],
    *,
    max_cases: int = _MAX_CASES,
) -> tuple[bool | None, list[BeatRow]]:
    """The measured gate: ``(beat, per-case rows)``.

    Only *registered* cases measure (the xcorpus sites are instance
    sources, not benchmarks): scripted-signal cases first, then
    strict-match cases in name order, up to *max_cases*.  ``beat`` is
    ``True`` when at least one case wins, ``None`` when no registered
    case exercises the candidate (the gate is unevaluable, not
    failed).
    """
    named = [c for c in ctxs if c.registered]
    signal = [c for c in named if candidate.name in c.fires]
    sig_ids = {id(c) for c in signal}
    rest = [
        c
        for c in named
        if id(c) not in sig_ids and _relevant(candidate, c)
    ]
    todo = (signal + rest)[:max_cases]
    rows = [_measure_case(candidate, c) for c in todo]
    if not rows or all(r.error for r in rows):
        return None, rows
    return bool(any(r.beat for r in rows)), rows


# ---------------------------------------------------------------------------
#  The sieve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SieveRow:
    """One candidate's three-gate record and verdict."""

    name: str
    family: str = ""
    provenance: str = ""
    verdict: str = ""  # review|known|refuted|uncertified|unpaid|unmeasured|no-instance
    instance_src: str = ""
    lhs_repr: str = ""
    rhs_repr: str = ""
    truth: str = ""  # KernelSink fp64 oracle outcome on the instance
    cert: bool = False
    cert_note: str = ""
    derivable: bool | None = None
    witness_rules: tuple[str, ...] = ()
    deriv_note: str = ""
    siblings: tuple[str, ...] = ()
    beat: bool | None = None
    beats: tuple[BeatRow, ...] = field(default_factory=tuple)
    measure_note: str = ""

    def to_dict(self) -> dict:
        """The JSON shape the queue doc and ``--json`` share."""
        return {
            "name": self.name,
            "family": self.family,
            "provenance": self.provenance,
            "verdict": self.verdict,
            "instance_src": self.instance_src,
            "truth": self.truth,
            "cert": self.cert,
            "derivable": self.derivable,
            "witness_rules": list(self.witness_rules),
            "deriv_note": self.deriv_note,
            "siblings": list(self.siblings),
            "beat": self.beat,
            "measure_note": self.measure_note,
            "lhs": self.lhs_repr,
            "rhs": self.rhs_repr,
            "cases": [
                {
                    "case": b.case,
                    "certified": b.certified,
                    "verified": b.verified,
                    "fired": b.fired,
                    "used": b.used,
                    "cost_before": b.cost_before,
                    "cost_after": b.cost_after,
                    "floor": b.floor,
                    "beat": b.beat,
                    "error": b.error,
                }
                for b in self.beats
            ],
        }


def sieve(
    candidate: Any,
    bank: list | None = None,
    *,
    ctxs: list[_CaseCtx] | None = None,
    measure: bool = True,
    max_cases: int = _MAX_CASES,
    measurer: Any = None,
) -> SieveRow:
    """Run the three gates on *candidate* against *bank*.

    ``bank`` defaults to :func:`known_bank` — ``KERNEL_RULES`` plus
    the substrate surface.  ``ctxs`` defaults to
    :func:`case_contexts` — the kernel corpus is pure spelled IR and
    cheap to build, so the instance search gets it even with
    ``measure=False``; the flag only gates the beat arm.  ``measurer``
    overrides the measured gate's callable ``(candidate, ctxs) ->
    (beat, rows)`` — the seam the test suite stubs.
    """
    bank = known_bank() if bank is None else list(bank)
    if ctxs is None:
        ctxs = case_contexts()
    measure_fn = measurer or measure_gate
    row_kw: dict[str, Any] = {
        "name": candidate.name,
        "family": _family(candidate.name),
        "provenance": law_provenance(candidate),
    }
    got = instance_of(candidate, ctxs)
    if got is None:
        return SieveRow(
            verdict="no-instance",
            measure_note="no instance: gates unevaluable",
            **row_kw,
        )
    src, (lhs, rhs) = got
    row_kw.update(
        instance_src=src, lhs_repr=op_repr(lhs), rhs_repr=op_repr(rhs)
    )

    # -- truth: fp64 KernelSink eval of the instance (an "unequal" is
    #    a measured counterexample and refutes outright; "skip"/
    #    "indeterminate" are inconclusive, never a refute)
    truth, _tnote = kernel_truth(lhs, rhs)
    row_kw["truth"] = truth
    if truth == "unequal":
        return SieveRow(verdict="refuted", **row_kw)

    # -- gate 1: cert — the candidate's own rule must carry a
    #    replayable certificate on its instance
    cres = verify_law(lhs, rhs, [candidate])
    row_kw["cert"] = bool(cres.derivable and cres.replayable)
    row_kw["cert_note"] = cres.note
    if not row_kw["cert"]:
        return SieveRow(verdict="uncertified", **row_kw)

    # -- gate 3 (before 2 — cheaper): derivability under bank minus
    #    alpha-siblings
    premises, sibs = _premises(candidate, bank)
    row_kw["siblings"] = sibs
    dres = verify_law(lhs, rhs, premises)
    row_kw["derivable"] = dres.derivable
    row_kw["witness_rules"] = dres.witness_rules
    row_kw["deriv_note"] = dres.note or dres.stop
    if dres.derivable:
        return SieveRow(verdict="known", **row_kw)

    # -- gate 2: measured pay on the registered kernel cases
    if measure:
        beat, beats = measure_fn(candidate, ctxs, max_cases=max_cases)
        row_kw["beat"] = beat
        row_kw["beats"] = tuple(beats)
        row_kw["measure_note"] = (
            f"{len(beats)} case(s) run"
            if beats
            else "no registered case exercises this law"
        )
    else:
        row_kw["beat"] = None
        row_kw["measure_note"] = "measurement skipped"

    if row_kw["beat"] is True:
        verdict = "review"
    elif row_kw["beat"] is False:
        verdict = "unpaid"
    else:
        verdict = "unmeasured"
    return SieveRow(verdict=verdict, **row_kw)


def sieve_all(
    candidates: list | None = None,
    bank: list | None = None,
    *,
    measure: bool = True,
    max_cases: int = _MAX_CASES,
    measurer: Any = None,
    progress: Any = None,
) -> list[SieveRow]:
    """Batch mode: sieve every candidate against *bank*.

    Corpus contexts are built once and shared across candidates.
    """
    bank = known_bank() if bank is None else list(bank)
    laws = shipped_candidates() if candidates is None else candidates
    ctxs = case_contexts()
    rows = []
    for c in laws:
        row = sieve(
            c,
            bank,
            ctxs=ctxs,
            measure=measure,
            max_cases=max_cases,
            measurer=measurer,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    return rows


# ---------------------------------------------------------------------------
#  The review-queue document
# ---------------------------------------------------------------------------


def _bool(b: bool | None) -> str:
    return {True: "yes", False: "no", None: "—"}[b]


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def render_markdown(
    rows: list[SieveRow], *, env: dict | None = None
) -> str:
    """The human document — one row per candidate, then the queue."""
    env = env or {}
    queue = [r for r in rows if r.verdict == "review"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    lines = [
        "# Kernel novelty sieve",
        "",
        "The codegen port of the novelty sieve — the automatic "
        "three-gate filter between a shipped ``KERNEL_RULES`` law and "
        "the human-review queue.  **cert** = the law's own rewrite "
        "carries a standalone-replayable certificate on its instance "
        "(`verify_law` + `verify_certificate`); **derivable** = the "
        "known bank — `KERNEL_RULES` U the shared substrate surface "
        "(`default_rules` + `ALL_RULES_WITH_LAYOUT` + `ATTENTION` + "
        "scan monoids + the prior-art FUSION/CATEGORICAL slice) — "
        "minus the candidate and its alpha-siblings, already proves "
        "the instance; **beat** = on some registered "
        "`bench.discovery.kernel` case the certified extraction under "
        "the full kernel bank lands strictly below the best "
        "extraction the rest of the bank reaches on the same term "
        "(the floor — itself never worse than spelled) AND the "
        "derivation certificate invokes the candidate.  A law is a "
        "novelty iff cert ∧ beat ∧ ¬derivable.  `truth` is the fp64 "
        "`KernelSink.verify` oracle on the instance — an `unequal` "
        "refutes outright.",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')} · "
        f"bank {env.get('bank_size', '?')} rules",
        "- verdicts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "| law | family | prov | instance | truth | cert | derivable | beat | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wit = (
            f" ({', '.join(r.witness_rules[:3])})"
            if r.derivable and r.witness_rules
            else ""
        )
        lines.append(
            f"| `{r.name}` | {r.family or '—'} | "
            f"{r.provenance or '—'} | {r.instance_src or '—'} "
            f"| {r.truth or '—'} | {_bool(r.cert)} "
            f"| {_bool(r.derivable)}{wit} | {_bool(r.beat)} "
            f"| **{r.verdict}** |"
        )
    lines += ["", "## Review queue", ""]
    if queue:
        lines.append(
            "Certified, measured-paying, and underivable from the "
            "known bank — these are for a human to look at:"
        )
        lines.append("")
        for r in queue:
            wins = ", ".join(
                f"{b.case} ({_num(b.cost_before)} → "
                f"{_num(b.cost_after)} vs floor {_num(b.floor)})"
                for b in r.beats
                if b.beat
            )
            lines.append(f"- `{r.name}` — {wins}")
    else:
        lines.append(
            "**Empty.** Every candidate either fails a gate or is a "
            "theorem of the bank — nothing currently needs a "
            "reviewer's eyes."
        )
    lines += ["", "## Per-law detail", ""]
    for r in rows:
        lines += [
            f"### `{r.name}` — {r.verdict}",
            "",
            f"- instance ({r.instance_src or 'none'}): "
            f"`{r.lhs_repr or '—'}` → `{r.rhs_repr or '—'}`",
            f"- truth: {r.truth or '—'} · cert: {_bool(r.cert)}"
            + (f" ({r.cert_note})" if r.cert_note else ""),
            f"- derivable: {_bool(r.derivable)}"
            + (
                f" via {', '.join(r.witness_rules)}"
                if r.witness_rules
                else ""
            )
            + (f" — {r.deriv_note}" if r.deriv_note else ""),
            f"- siblings excluded: {', '.join(r.siblings) or 'none'}",
            f"- beat: {_bool(r.beat)} — {r.measure_note or '—'}",
        ]
        for b in r.beats:
            if b.error:
                lines.append(f"  - `{b.case}`: error — {b.error}")
                continue
            lines.append(
                f"  - `{b.case}`: cert {_bool(b.certified)}, "
                f"verify {_bool(b.verified)}, fired {b.fired}, "
                f"in-deriv {_bool(b.used)}, cost "
                f"{_num(b.cost_before)} → {_num(b.cost_after)} "
                f"vs floor {_num(b.floor)} → beat {_bool(b.beat)}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _env() -> dict:
    """The provenance block — timestamp, git sha, torch, bank size."""
    import datetime as _dt

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return ""

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
        "bank_size": len(known_bank()),
    }


def main(argv: list[str] | None = None) -> int:
    """Sieve ``KERNEL_RULES``; emit the review-queue document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--names",
        help="comma-separated KERNEL_RULES names to sieve "
        "(default: the whole bank)",
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help="skip the measured gate (cert + derivability only)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=_MAX_CASES,
        help="per-candidate cap on measured cases",
    )
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument(
        "--out",
        default=str(DOC_PATH),
        help=f"markdown output (default {DOC_PATH})",
    )
    args = parser.parse_args(argv)

    bank = known_bank()
    if args.names:
        pool = {r.name: r for r in KERNEL_RULES}
        laws = []
        for n in args.names.split(","):
            n = n.strip()
            if n in pool:
                laws.append(pool[n])
            else:
                print(f"unknown law {n!r}", file=sys.stderr)
                return 2
    else:
        laws = shipped_candidates()

    print(  # stdout-compat
        f"sieving {len(laws)} candidates against a "
        f"{len(bank)}-rule bank"
    )
    rows = sieve_all(
        laws,
        bank,
        measure=not args.no_measure,
        max_cases=args.max_cases,
        progress=lambda r: print(  # stdout-compat
            f"  {r.name:<34} {r.verdict}", flush=True
        ),
    )

    env = _env()
    md = render_markdown(rows, env=env)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")  # stdout-compat

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "env": env,
                    "candidates": [r.to_dict() for r in rows],
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(f"wrote {args.json}")  # stdout-compat

    queue = [r.name for r in rows if r.verdict == "review"]
    print(f"review queue: {queue or 'empty'}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
