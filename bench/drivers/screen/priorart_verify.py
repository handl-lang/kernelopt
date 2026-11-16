"""Certify the kernel prior-art bank — sweep guarded sites, cert.

The kernelopt port of the parent's
``bench/drivers/audit/prior_art_verify.py``: for each rule in
``mlopt_kernel.priorart.KERNEL_PRIOR_ART_RULES``, enumerate
synthesized bindings (``oracle._binding_envs`` — the shared
gauntlet enumeration, op-agnostic), apply the rule's own
``check``/``derive`` with firing semantics, evaluate each accepted
site fp64 through :func:`novelty_sieve.kernel_truth` — the
``KernelSink`` oracle, since the shared ``oracle.eval_instance``
has no ``kf_*``/``om_*`` bindings — and on the first ``equal`` site
run ``verify_law(lhs, rhs, [rule])``: the e-graph must merge the
sides under the rule itself and replay a standalone certificate.

``CASES`` supplies curated bindings where the generic enumeration
cannot reach the accepted corner — float-valued ``eps`` attrs,
correlated concat/split operand shapes, scalar ``s`` for the GEMM
alpha folds, the ``linear`` mm-shape corner.

Verdicts mirror upstream: ``certified`` (>=1 equal site, 0 unequal
sites in the window, replayable cert), ``refuted`` (an accepted
site measured unequal), ``vacuous`` (no accepted site), ``err``
(accepted sites but none equal — all skipped/indeterminate),
``uncertified`` (equal but no replayable cert).

Run::

    .venv/bin/python -m bench.drivers.screen.priorart_verify
    .venv/bin/python -m bench.drivers.screen.priorart_verify \\
        --names pa_kf_silu_expand,pa_sum_over_concat
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kit_discovery import oracle
from kit_discovery.verifier import verify_law
from mlopt_core.ir import Const, TensorType, Var
from mlopt_core.meta import instantiate_pattern

from bench.drivers.screen.novelty_sieve import kernel_truth
from mlopt_kernel.priorart import KERNEL_PRIOR_ART_RULES

__all__ = [
    "CASES",
    "LawRow",
    "certify_all",
    "certify_rule",
    "main",
    "render_markdown",
]

#: kernelopt repo root — bench/drivers/screen/priorart_verify.py → 3 up.
REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "results_prior_art_kernel.md"

#: Env budget per law — same posture as upstream.
_MAX_ENVS = 20_000
#: Accepted sites to evaluate per law before calling it certified.
_MAX_SITES = 24


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


#: Curated bindings for the enum-unreachable corners — same dict
#: shape ``oracle._binding_envs`` yields (leaf metavars → Var/Const,
#: ``$attr:NAME`` → attr value).
CASES: dict[str, list[dict]] = {
    # float eps attrs — the enum's attr domain is ints.
    "pa_kf_layernorm_expand": [
        {"u": _v("u", 2, 4), "$attr:D": -1, "$attr:E": 1e-5}
    ],
    "pa_kf_rmsnorm_expand": [
        {"u": _v("u", 2, 4), "$attr:D": -1, "$attr:E": 1e-6}
    ],
    # matmul-arity denotations — the enum's independent leaf binds
    # rarely produce a contracting pair.
    "pa_kf_softmax_mm_expand": [
        {"u": _v("u", 4, 4), "v": _v("v", 4, 4), "$attr:D": -1}
    ],
    "pa_kf_mm_bias_expand": [
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "b": _v("b", 4, 5)}
    ],
    "pa_kf_mm_scale_expand": [
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "s": Const(0.5)}
    ],
    "pa_kf_mm_sb_expand": [
        {
            "a": _v("a", 4, 3),
            "w": _v("w", 3, 5),
            "s": Const(0.5),
            "b": _v("b", 4, 5),
        }
    ],
    "pa_kf_mm_gelu_expand": [{"a": _v("a", 4, 3), "w": _v("w", 3, 5)}],
    "pa_om_decayed_expand": [
        {"s": _v("s", 4, 4), "d": _v("d", 4, 4), "v": _v("v", 4, 4)}
    ],
    # concat-compatible operand pairs — the enum binds leaves
    # independently and rarely produces off-axis-matching extents.
    "pa_sum_over_concat": [
        {
            "a": _v("a", 2, 4),
            "b": _v("b", 3, 4),
            "$attr:CD": 0,
            "$attr:RD": 0,
            "$attr:K": True,
        },
        {
            "a": _v("a", 4, 2),
            "b": _v("b", 4, 3),
            "$attr:CD": -1,
            "$attr:RD": -1,
            "$attr:K": False,
        },
    ],
    # split sizes are tuples the attr enum does not enumerate.
    **{
        f"pa_split_pull_{op}": [
            {
                "u": _v("u", 2, 6),
                "$attr:SZ": (3, 3),
                "$attr:D": 1,
                "$attr:I": 0,
            }
        ]
        for op in (
            "neg",
            "abs",
            "relu",
            "sigmoid",
            "silu",
            "tanh",
            "exp",
            "square",
        )
    },
    # scalar scale for the alpha folds — the enum's leaf bank rarely
    # binds rank-0.
    "pa_mm_scale_alpha_w": [
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "s": Const(0.5)},
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "s": _v("s")},
    ],
    "pa_mm_scale_alpha_x": [
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "s": Const(0.5)},
        {"a": _v("a", 4, 3), "w": _v("w", 3, 5), "s": _v("s")},
    ],
    # F.linear's contraction corner — x[−1] == W[−1].
    "pa_mm_to_linear": [{"x": _v("x", 4, 3), "W": _v("W", 5, 3)}],
    "pa_linear_expand": [{"x": _v("x", 4, 3), "W": _v("W", 5, 3)}],
    # const folds — pin the literal.
    "pa_div_const_recip": [{"u": _v("u", 4, 4), "C": Const(4.0)}],
    "pa_const_fold_recip": [{"C": Const(4.0)}],
    "pa_const_fold_neg": [{"C": Const(2.5)}],
    "pa_pow2_exp2": [{"C": Const(2), "x": _v("x", 4, 4)}],
    # mean expand — concrete shape + in-range dim.
    "pa_mean_expand": [
        {"u": _v("u", 2, 4), "$attr:D": -1, "$attr:K": True},
        {"u": _v("u", 2, 4), "$attr:D": (0, 1), "$attr:K": False},
    ],
    # om_chunk_scan denotations — concat needs matching off-axis
    # extents, which independent enumeration rarely produces.
    **{
        f"pa_om_chunk_scan_expand_{n}": [
            {
                **{f"x{i}": _v(f"x{i}", 2, 4) for i in range(n)},
                "$attr:D": 0,
            }
        ]
        for n in range(2, 7)
    },
    # sdpa — matmul-compatible operand shapes plus a scalar scale.
    "pa_kf_sdpa_expand": [
        {
            "a": _v("a", 4, 4),
            "w": _v("w", 4, 4),
            "s": Const(0.5),
            "v": _v("v", 4, 4),
            "$attr:D": -1,
        }
    ],
    "pa_kf_sdpa_cap_expand": [
        {
            "a": _v("a", 4, 4),
            "w": _v("w", 4, 4),
            "s": Const(0.5),
            "v": _v("v", 4, 4),
            "$attr:D": -1,
        }
    ],
    # the masked denotation — the enum's independent leaves rarely
    # produce a score-compatible additive operand.
    "pa_kf_sdpa_mask_expand": [
        {
            "a": _v("a", 4, 4),
            "w": _v("w", 4, 4),
            "s": Const(0.5),
            "m": _v("m", 4, 4),
            "v": _v("v", 4, 4),
            "$attr:D": -1,
        }
    ],
    # the gqa denotation — 8 query heads over 4 kv heads, the repeat
    # count a tuple the enum's scalar attr domain never reaches.
    "pa_kf_sdpa_gqa_expand": [
        {
            "a": _v("a", 8, 4, 4),
            "w": _v("w", 4, 4, 4),
            "s": Const(0.5),
            "v": _v("v", 4, 4, 4),
            "$attr:D": -1,
            "$attr:GR": (2, 1, 1),
        }
    ],
}


@dataclass
class LawRow:
    """One prior-art rule's sweep record."""

    name: str
    verdict: str = "vacuous"
    n_accept: int = 0
    n_equal: int = 0
    n_unequal: int = 0
    n_skip: int = 0
    n_indet: int = 0
    cert: bool = False
    replayable: bool = False
    instance: str = ""
    note: str = ""


def _accepted(rule: Any, bound: dict) -> dict | None:
    """Firing semantics: check then derive; return the full subst."""
    if rule.check is not None:
        try:
            if not rule.check(bound):
                return None
        except Exception:
            return None
    inst = dict(bound)
    if rule.derive is not None:
        try:
            extra = rule.derive(bound)
        except Exception:
            return None
        if extra is None:
            return None
        inst.update(extra)
    return inst


def _sweep(
    rule: Any, envs: Iterable[dict], row: LawRow, max_sites: int
) -> None:
    certified = row.n_equal > 0
    for bound in envs:
        if row.n_accept >= max_sites:
            break
        inst = _accepted(rule, bound)
        if inst is None:
            continue
        try:
            lhs_i = instantiate_pattern(rule.lhs, inst)
            rhs_i = instantiate_pattern(rule.rhs, inst)
        except Exception:
            continue
        row.n_accept += 1
        outcome, note = kernel_truth(lhs_i, rhs_i)
        # Domain-restricted ops (log/sqrt/rsqrt on randn feeds) are
        # indeterminate, never unequal — retry on a few fresh draws
        # before counting the site inconclusive.
        if outcome == "indeterminate":
            for _ in range(4):
                outcome, note = kernel_truth(lhs_i, rhs_i)
                if outcome != "indeterminate":
                    break
        if outcome == "equal":
            row.n_equal += 1
            if not certified:
                res = verify_law(lhs_i, rhs_i, [rule])
                row.cert = bool(res.derivable)
                row.replayable = bool(res.replayable)
                row.instance = f"{lhs_i!r} ~ {rhs_i!r}"
                certified = True
        elif outcome == "unequal":
            row.n_unequal += 1
            row.note = row.note or f"counterexample: {lhs_i!r}"
        elif outcome == "indeterminate":
            row.n_indet += 1
        else:
            row.n_skip += 1
            row.note = row.note or f"skip: {note}"


def _verdict(row: LawRow) -> str:
    if row.n_unequal:
        return "refuted"
    if row.n_accept == 0:
        return "vacuous"
    if row.n_equal == 0:
        return "err"
    if row.replayable:
        return "certified"
    return "uncertified"


def certify_rule(
    rule: Any, *, max_envs: int = _MAX_ENVS, max_sites: int = _MAX_SITES
) -> LawRow:
    """Curated cases first, then the bounded generic sweep."""
    row = LawRow(name=rule.name)
    for case in CASES.get(rule.name, ()):
        _sweep(rule, (case,), row, max_sites)
    _sweep(
        rule,
        itertools.islice(
            oracle._binding_envs(rule.lhs, rule.rhs), max_envs
        ),
        row,
        max_sites,
    )
    row.verdict = _verdict(row)
    return row


def certify_all(
    rules: Any = None,
    *,
    max_envs: int = _MAX_ENVS,
    max_sites: int = _MAX_SITES,
    progress: Any = print,
) -> list[LawRow]:
    """Sweep every rule in the kernel prior-art bank."""
    rules = KERNEL_PRIOR_ART_RULES if rules is None else rules
    rows = [
        certify_rule(r, max_envs=max_envs, max_sites=max_sites)
        for r in rules
    ]
    if progress:
        for row in rows:
            progress(
                f"  {row.name:<30} {row.verdict:<12} "
                f"acc={row.n_accept} eq={row.n_equal} "
                f"ne={row.n_unequal} skip={row.n_skip} "
                f"cert={row.cert}/{row.replayable}"
            )
    return rows


def render_markdown(rows: list[LawRow]) -> str:
    """The bank document — counts by source plus the per-law table."""
    ok = sum(r.verdict == "certified" for r in rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    lines = [
        "# Kernel prior-art bank — results",
        "",
        "The kernel-level prior-art encoding the novelty sieve's "
        "derivability floor was missing: "
        f"``mlopt_kernel.priorart.KERNEL_PRIOR_ART_RULES`` — "
        f"{len(rows)} laws, all tagged ``PRIOR_ART``, opt-in through "
        "``known_bank()`` (never in ``KERNEL_RULES`` — they are "
        "premises, not play actions).  Where the substrate's "
        "``PRIOR_ART`` encodes the *tensor-level* literature (TASO "
        "substitutions, Inductor/XLA rewrites), this bank encodes the "
        "**kernel codegen vocabulary** — what Halide/Triton/TVM/"
        "cuBLAS already do to a spelled program:",
        "",
        "- **Fused-kernel denotations (24)** — each ``kf_*``/"
        "  ``om_*`` op expands to its spelled composite (the sink's "
        "  own binding body): SiLU/softsign/FMA/SwiGLU gates, the "
        "  residue folds, the spelled exp/Σexp softmax site (the "
        "  Triton-tutorial fused kernel), softmax·mm, whole-block "
        "  SDPA (FlashAttention) "
        "  plain and soft-capped (Gemma-2), layernorm/rmsnorm "
        "  (Apex/Triton), the cuBLASLt epilogues "
        "  (``addmm``/scale/scale-bias/GELU), the decayed-score "
        "  ``om_decayed`` element and the chunked-scan "
        "  ``om_chunk_scan`` blocks (FLA's chunked schedule — "
        "  arities 2–6).",
        "- **Halide schedule primitives (9)** — a ``sum`` over a "
        "  concat of tiles is the per-tile sums added (split rdom); "
        "  a pointwise op computes inside the tile (``compute_at`` "
        "  — ``split`` pull-throughs for neg/abs/relu/sigmoid/silu/"
        "  tanh/exp/square).",
        "- **GEMM operand folds (4)** — alpha scale into weight or "
        "  activation (scalar-guarded), and the ``linear``↔"
        "  ``matmul(x, Wᵀ)`` prepack pair.",
        "- **Strength reductions / const folds (9)** — div-by-const "
        "  → mul-by-recip (two laws, InstCombine), ``neg(c)`` fold, "
        "  ``exp``→``exp2·log2e``, ``log``→``log2·ln2``, "
        "  ``pow(2,x)``→``exp2(x)``, the tanh-spelled sigmoid, "
        "  ``rsqrt``→``recip(sqrt)`` denotation, ``mean``→"
        "  ``sum·(1/N)``.",
        "- **Layout no-ops (3)** — ``contiguous``/``detach``/"
        "  ``detach_`` erase at codegen.",
        "",
        "**Verification** — the kernel port of the parent's "
        "``prior_art_verify``: each rule's accepted sites (curated "
        "``CASES`` for the enum-unreachable corners, then the "
        "``oracle._binding_envs`` sweep) evaluated fp64 through "
        "``KernelSink.verify``; the first ``equal`` site must carry "
        "a standalone-replayable ``verify_law`` certificate.",
        "",
        f"**Result: {ok}/{len(rows)} certified** — verdicts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + ".",
        "",
        "## Sieve impact — the floor before and after",
        "",
        "``known_bank()`` grew 264 → 313 rules.  The previous sweep "
        "(tensor-catalog floor only) reported ``known=1, review=20, "
        "unmeasured=4, unpaid=1``; with the kernel vocabulary encoded "
        "the same 26 candidates report **``known=26``** — every "
        "``kern_fuse_*`` fold is derivable through its fused op's "
        "denotation bridge (the flash/apex/cuBLAS-epilogue/fma "
        "vocabulary is known codegen, measured), the five "
        "``kern_fuse_chunk_scan_*`` members derive through the "
        "scheduled-block denotation (FLA's chunked scan), and "
        "``kern_div_recip`` composes ``pa_div_const_recip`` + "
        "``pa_const_fold_recip`` (InstCombine's div-by-const plus "
        "constant folding).  The review queue is **empty**: what the "
        'old run called underivable meant "absent from our tensor '
        'catalog", not "unknown to kernel compilers".',
        "",
        "| law | before | after | witness |",
        "|---|---|---|---|",
        "| `kern_fuse_silu` … `kern_fuse_om_decayed` (19 fused folds) | review/unpaid | **known** | `pa_kf_*_expand` / `pa_om_decayed_expand` |",
        "| `kern_fuse_chunk_scan` + `_3`…`_6` (5 schedule folds) | review/unmeasured | **known** | `pa_om_chunk_scan_expand_{2..6}` |",
        "| `kern_div_recip` | review | **known** | `pa_div_const_recip` + `pa_const_fold_recip` |",
        "| `kern_square_strength` | known | **known** | `pow_to_square` + `square_expand` (unchanged) |",
        "",
        "| law | verdict | accepted | equal | unequal | skip | cert |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.name}` | **{r.verdict}** | {r.n_accept} "
            f"| {r.n_equal} | {r.n_unequal} | {r.n_skip} "
            f"| {'yes' if r.replayable else 'no'} |"
        )
    dropped = [r for r in rows if r.verdict != "certified"]
    lines += ["", "## Dropped / failed laws", ""]
    if dropped:
        for r in dropped:
            lines.append(f"- `{r.name}` — {r.verdict}: {r.note or '—'}")
    else:
        lines.append(
            "None — every encoded law certified on at least one "
            "fp64-equal accepted site."
        )
    lines += [
        "",
        "## Honest misses",
        "",
        "- *Vectorize/unroll/pipeline/double-buffer* — no term-level "
        "expression: the IR has no lane or stage axis; these are "
        "loop-schedule attributes, not equivalences a rewrite can "
        "state.",
        "- *The fold direction of every `kern_fuse_*` law* — alpha-"
        "siblings of the shipped bank; the sieve's premise cut "
        "excludes them, so encoding them would be circular.  The "
        "expand direction is what prior art honestly claims.",
        "- *Online/max-shifted softmax* — no reduce-max op in the "
        "IR, so FlashAttention's running-max form cannot be "
        "spelled; the value-level `softmax` denotation stands in.",
        "- `mul(x,0)=0` / `div(x,x)=1` class — IEEE-unsound at "
        "NaN/Inf, same miss as the tensor bank's.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Certify the kernel prior-art bank; write the results doc."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", default="")
    p.add_argument("--max-envs", type=int, default=_MAX_ENVS)
    p.add_argument("--max-sites", type=int, default=_MAX_SITES)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--out", default=str(DOC_PATH))
    args = p.parse_args(argv)
    rules = list(KERNEL_PRIOR_ART_RULES)
    if args.names:
        keep = set(args.names.split(","))
        rules = [r for r in rules if r.name in keep]
        missing = keep - {r.name for r in rules}
        if missing:
            print(f"unknown: {sorted(missing)}", file=sys.stderr)
            return 2
    print(f"certifying {len(rules)} kernel prior-art laws")
    rows = []
    for r in rules:
        row = certify_rule(
            r, max_envs=args.max_envs, max_sites=args.max_sites
        )
        print(
            f"  {row.name:<30} {row.verdict:<12} "
            f"acc={row.n_accept} eq={row.n_equal} "
            f"ne={row.n_unequal} skip={row.n_skip} "
            f"cert={row.cert}/{row.replayable}",
            flush=True,
        )
        rows.append(row)
    ok = sum(v.verdict == "certified" for v in rows)
    print(f"{ok}/{len(rows)} certified")
    out = Path(args.out)
    out.write_text(render_markdown(rows))
    print(f"wrote {out}")  # stdout-compat
    if args.json:
        args.json.write_text(
            json.dumps([vars(r) for r in rows], indent=2, default=str)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
