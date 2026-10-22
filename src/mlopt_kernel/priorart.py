# ruff: noqa: RUF001, RUF003 — math notation (≡, σ, −, ·) in docstrings.
"""Kernel-level prior-art bank — the known codegen vocabulary as laws.

The kernelopt port of ``mlopt_core.laws.priorart``: where that bank
encodes the *tensor-level* transform literature (TASO's substitution
set, the Inductor/XLA rewrite surface), this one encodes what the
**kernel compilers** already own — the codegen vocabulary a spelled
program gets rewritten into before it ever reaches a dispatcher:

1. **Fused-kernel denotations** — every ``kf_*``/``om_*`` op expands
   to its spelled composite body (the same composite the
   ``KernelSink`` binding table compiles).  A fused SiLU/FMA/softmax/
   layernorm/RMSNorm kernel is a Triton tutorial; the matmul
   epilogues are cuBLASLt's ``CUBLASLT_EPILOGUE_*``; the whole-block
   SDPA fold is FlashAttention; the decayed-mask element and the
   chunked-scan block are the FLA chunked schedule.  Encoding the
   *denotation* (fused op → spelled work) is what makes "fold to the
   fused kernel" a theorem of the known bank rather than a novelty:
   the fold direction itself is each ``kern_fuse_*`` law's alpha-
   sibling and would be excluded from the premise bank — the expand
   direction is the honest prior-art statement.
2. **Halide schedule primitives** as far as the term IR reaches:
   tiling distributions (a reduction over a concat of tiles is the
   sum of the per-tile reductions; a pointwise op computes inside
   the tile — ``compute_at``/inline producer fusion) and the
   scheduled-scan block's spelled denotation.
3. **GEMM operand folds** — the scale-into-operand and
   transpose-prepack moves cuBLAS's alpha/beta and ``F.linear``'s
   weight layout already perform.
4. **Strength reductions and constant folding on kernels** — the
   InstCombine-class moves a device code generator emits: div-by-
   constant → multiply-by-reciprocal, ``exp`` → ``exp2``·log2e,
   ``pow(2,·)`` → ``exp2``, the tanh-spelled sigmoid, ``rsqrt``'s
   reciprocal-root denotation, ``mean`` = ``sum``·(1/N).
5. **Layout no-ops** — ``contiguous``/``detach`` erase at codegen.

What is deliberately NOT here — the honest miss list:

* *Vectorization/unrolling/pipelining* — no term-level expression:
  the IR has no lane/vector or stage axis.  Halide's
  ``vectorize``/``unroll``/``prefetch`` are schedule attributes on a
  loop, not program equivalences a rewrite can state.
* *The fold direction of every ``kern_fuse_*`` law* — those are
  alpha-siblings of the shipped bank and the sieve's premise cut
  excludes them by construction; encoding them would be circular.
* ``mul(x,0)=0``-class identities — IEEE-unsound at NaN/Inf, same as
  the tensor bank's miss list.
* *Online/max-shifted softmax* — the IR has no reduce-max op, so
  FlashAttention's running-max form cannot be spelled; the
  value-level ``softmax`` denotation is what gets encoded instead.

Every rule is tagged ``PRIOR_ART`` (plus a structural tag) and lives
outside ``KERNEL_RULES`` — the bank composes through
``bench.drivers.screen.novelty_sieve.known_bank`` only, the same
opt-in posture as the substrate's ``prior_art`` preset.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op
from mlopt_core.laws import tags
from mlopt_core.laws.base import R, _shape_of

__all__ = ["KERNEL_PRIOR_ART_RULES"]

_PA = (tags.PRIOR_ART,)
_PA_SIM = (tags.PRIOR_ART, tags.SIMPLIFICATION)
_PA_FUS = (tags.PRIOR_ART, tags.FUSION)
_PA_CAT = (tags.PRIOR_ART, tags.CATEGORICAL)


# ---------------------------------------------------------------------------
#  Shared guards
# ---------------------------------------------------------------------------


def _scalar(bound: dict) -> bool:
    """``s`` bound to a scalar — a ``Const`` literal or a rank-0 leaf.

    The GEMM alpha-folds commute the product with a scalar scale
    only; a non-scalar scale would need the broadcast spelled into
    the weight's rows, which is a different (shape-dependent) law.
    """
    t = bound.get("s")
    if isinstance(t, Const):
        return isinstance(t.value, (int, float)) and not isinstance(
            t.value, bool
        )
    s = _shape_of(t)
    return isinstance(s, tuple) and s == ()


def _mean_count(bound: dict) -> dict | None:
    """``1/N`` leaf for ``mean(u, dim=D)`` — N the reduced extent."""
    s = _shape_of(bound.get("u"))
    d = bound.get("$attr:D")
    if not (isinstance(s, tuple) and s):
        return None
    dims = d if isinstance(d, (tuple, list)) else (d,)
    n = 1
    for dd in dims:
        if not isinstance(dd, int):
            return None
        ext = s[dd % len(s)]
        if not isinstance(ext, int):
            return None
        n *= ext
    if n == 0:
        return None
    return {"N": Const(1.0 / n)}


def _mean_guard(bound: dict) -> bool:
    return _mean_count(bound) is not None


# ---------------------------------------------------------------------------
#  §1 — Fused-kernel denotations: the compiled composite IS the spelled body
#
#  Each law expands one fused/scheduled op to the composite the sink's
#  binding table compiles — the published kernel semantics (Triton
#  tutorials' fused softmax/layernorm, xformers' SwiGLU, cuBLASLt's
#  epilogues, FlashAttention's fused SDPA, FLA's chunked decayed
#  scan).  The direction is expand: the fold direction is the shipped
#  ``kern_fuse_*`` law (and its alpha-sibling — premise-excluded).
# ---------------------------------------------------------------------------

pa_kf_silu_expand = R(
    "pa_kf_silu_expand",
    Op.make("kf_silu", "a", "b"),
    Op.make("mul", "a", Op.make("sigmoid", "b")),
    law="A fused gate kernel denotes a·σ(b) — the elementwise pair "
    "Inductor inlines into one launch (SiLU is the a ≡ b section).",
    tags=_PA_FUS,
)

pa_kf_softsign_expand = R(
    "pa_kf_softsign_expand",
    Op.make("kf_softsign", "a", "b", "c"),
    Op.make("div", "a", Op.make("add", Op.make("abs", "b"), "c")),
    law="A fused softsign kernel denotes a/(|b|+c) — abs, add and "
    "div in a single elementwise pass.",
    tags=_PA_FUS,
)

pa_kf_addmul_expand = R(
    "pa_kf_addmul_expand",
    Op.make("kf_addmul", "a", "b", "c"),
    Op.make("add", Op.make("mul", "a", "b"), "c"),
    law="A fused multiply-add kernel denotes a·b + c — the hardware "
    "FMA (InstCombine's fmuladd formation, every device ISA's "
    "fused op).",
    tags=_PA_FUS,
)

pa_kf_swiglu_expand = R(
    "pa_kf_swiglu_expand",
    Op.make("kf_swiglu", "a", "b"),
    Op.make("mul", Op.make("mul", "a", Op.make("sigmoid", "a")), "b"),
    law="A fused SwiGLU kernel denotes (a·σ(a))·b — the LLaMA-class "
    "MLP gate the kernel libraries already fuse (xformers' "
    "swiglu kernel).",
    tags=_PA_FUS,
)

pa_kf_silu_sigmoid_expand = R(
    "pa_kf_silu_sigmoid_expand",
    Op.make("kf_silu_sigmoid", "a", "b"),
    Op.make(
        "mul",
        Op.make("mul", "a", Op.make("sigmoid", "a")),
        Op.make("sigmoid", "b"),
    ),
    law="A fused double-gate kernel denotes (a·σ(a))·σ(b) — an "
    "elementwise triple a codegen fuses whole.",
    tags=_PA_FUS,
)

pa_kf_addmul_tanh_expand = R(
    "pa_kf_addmul_tanh_expand",
    Op.make("kf_addmul_tanh", "a", "b", "c"),
    Op.make("add", Op.make("mul", "a", "b"), Op.make("tanh", "c")),
    law="A fused product-plus-tanh kernel denotes a·b + tanh(c) — "
    "the pointwise survivor absorbed into the epilogue, the generic "
    "elementwise-fusion move.",
    tags=_PA_FUS,
)

pa_kf_addmul2_expand = R(
    "pa_kf_addmul2_expand",
    Op.make("kf_addmul2", "a", "b", "c", "d"),
    Op.make("add", Op.make("mul", "a", "b"), Op.make("mul", "c", "d")),
    law="A fused two-product kernel denotes a·b + c·d — one "
    "elementwise pass over two FMAs.",
    tags=_PA_FUS,
)

pa_kf_mul3_expand = R(
    "pa_kf_mul3_expand",
    Op.make("kf_mul3", "a", "b", "c"),
    Op.make("mul", Op.make("mul", "a", "b"), "c"),
    law="A fused triple-product kernel denotes (a·b)·c — three "
    "elementwise multiplies in one pass.",
    tags=_PA_FUS,
)


# Softmax/normalization: the fused kernel denotes the spelled
# exp/sum/div normalization — the same composite the sink compiles
# (``softmax_fold`` in the default bank already equates the spelled
# site with the library ``softmax`` op, so both readings coexist in
# the e-class).
def _sm_site(score: Any) -> Op:
    """``div(exp(score), Σexp(score), dim=D, keepdim=True)``."""
    return Op.make(
        "div",
        Op.make("exp", score),
        Op.make("sum", Op.make("exp", score), dim="D", keepdim=True),
    )


pa_kf_softmax_expand = R(
    "pa_kf_softmax_expand",
    Op.make("kf_softmax", "u", dim="D"),
    _sm_site("u"),
    law="A fused softmax kernel denotes exp(u)/Σexp(u) — the "
    "Triton-tutorial fused normalization (exp/sum/div in one pass; "
    "``softmax_fold`` in the substrate bank equates the same site "
    "with the library op).",
    tags=_PA_FUS,
)

pa_kf_softmax_mm_expand = R(
    "pa_kf_softmax_mm_expand",
    Op.make("kf_softmax_mm", "u", "v", dim="D"),
    Op.make("matmul", _sm_site("u"), "v"),
    law="A fused softmax·matmul kernel denotes softmax(u)·v — the "
    "two-step attention kernel (pre-flash fused-softmax-then-GEMM).",
    tags=_PA_FUS,
)

pa_kf_sdpa_expand = R(
    "pa_kf_sdpa_expand",
    Op.make("kf_sdpa", "a", "w", "s", "v", dim="D", cap=False),
    Op.make(
        "matmul",
        _sm_site(Op.make("mul", Op.make("matmul", "a", "w"), "s")),
        "v",
    ),
    law="A fused SDPA kernel denotes softmax(qk·s)·v — Flash-"
    "Attention's single-pass attention (qk matmul, scale, softmax "
    "and value matmul in one kernel).",
    tags=_PA_FUS,
)

pa_kf_sdpa_cap_expand = R(
    "pa_kf_sdpa_cap_expand",
    Op.make("kf_sdpa", "a", "w", "s", "v", dim="D", cap=True),
    Op.make(
        "matmul",
        _sm_site(
            Op.make(
                "tanh",
                Op.make("mul", Op.make("matmul", "a", "w"), "s"),
            )
        ),
        "v",
    ),
    law="The soft-capped SDPA kernel denotes softmax(tanh(qk·s))·v "
    "— the Gemma-2 logit-cap spelling under the same one-pass "
    "schedule.",
    tags=_PA_FUS,
)


def _ln_core() -> Op:
    """``sub(u, mean(u, dim=D, keepdim=True))`` — the centred term."""
    return Op.make(
        "sub", "u", Op.make("mean", "u", dim="D", keepdim=True)
    )


pa_kf_layernorm_expand = R(
    "pa_kf_layernorm_expand",
    Op.make("kf_layernorm", "u", dim="D", eps="E"),
    Op.make(
        "mul",
        _ln_core(),
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", _ln_core(), Const(2)),
                    dim="D",
                    keepdim=True,
                ),
                "EPS",
            ),
        ),
    ),
    dspec={"$leaf:EPS": ("attr", "E")},
    law="A fused layernorm kernel denotes (u−μ)·rsqrt(mean((u−μ)²)"
    "+eps) — the Apex/Triton-tutorial fused norm (sub, two means, "
    "pow, add, rsqrt in one pass over the row).",
    tags=_PA_FUS,
)

pa_kf_rmsnorm_expand = R(
    "pa_kf_rmsnorm_expand",
    Op.make("kf_rmsnorm", "u", dim="D", eps="E"),
    Op.make(
        "mul",
        "u",
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", "u", Const(2)),
                    dim="D",
                    keepdim=True,
                ),
                "EPS",
            ),
        ),
    ),
    dspec={"$leaf:EPS": ("attr", "E")},
    law="A fused RMSNorm kernel denotes u·rsqrt(mean(u²)+eps) — "
    "the LLaMA-class norm kernel (mul, mean, pow, add, rsqrt in "
    "one pass).",
    tags=_PA_FUS,
)

pa_kf_mm_bias_expand = R(
    "pa_kf_mm_bias_expand",
    Op.make("kf_mm_bias", "a", "w", "b"),
    Op.make("add", Op.make("matmul", "a", "w"), "b"),
    law="A GEMM-with-bias kernel denotes (a@w)+b — the ``addmm`` "
    "epilogue cuBLAS/cuBLASLt absorb into the contraction.",
    tags=_PA_FUS,
)

pa_kf_mm_scale_expand = R(
    "pa_kf_mm_scale_expand",
    Op.make("kf_mm_scale", "a", "w", "s"),
    Op.make("mul", Op.make("matmul", "a", "w"), "s"),
    law="A scaled-GEMM kernel denotes (a@w)·s — cuBLAS's alpha "
    "scale on the output tile.",
    tags=_PA_FUS,
)

pa_kf_mm_sb_expand = R(
    "pa_kf_mm_sb_expand",
    Op.make("kf_mm_sb", "a", "w", "s", "b"),
    Op.make(
        "add", Op.make("mul", Op.make("matmul", "a", "w"), "s"), "b"
    ),
    law="A scale+bias GEMM kernel denotes (a@w)·s+b — the alpha-"
    "scale-plus-bias epilogue (the attention-scale spelling).",
    tags=_PA_FUS,
)

pa_kf_mm_gelu_expand = R(
    "pa_kf_mm_gelu_expand",
    Op.make("kf_mm_gelu", "a", "w"),
    Op.make("gelu", Op.make("matmul", "a", "w")),
    law="A GEMM+GELU kernel denotes gelu(a@w) — cuBLASLt's "
    "CUBLASLT_EPILOGUE_GELU; the transformer-MLP activation "
    "absorbed into the contraction.",
    tags=_PA_FUS,
)

pa_om_decayed_expand = R(
    "pa_om_decayed_expand",
    Op.make("om_decayed", "s", "d", "v"),
    Op.make("matmul", Op.make("mul", "s", "d"), "v"),
    law="The decayed-score kernel denotes (s ⊙ d) @ v — the "
    "retnet/FLA masked-decay elementwise-into-contraction fold "
    "(the decay mask folds into the score prologue).",
    tags=_PA_FUS,
)


def _chunk_scan_expand(n: int) -> Rewrite:
    """``om_chunk_scan(x₁..xₙ, dim) → concat(x₁..xₙ, dim)`` — arity n.

    The scheduled block's spelled denotation: the chunked decayed
    scan (FLA's ``chunked_delta_rule`` / Mamba's chunked form)
    computes exactly the per-chunk concat it schedules — the
    schedule is a batched plan, not a new function.
    """
    xs = [f"x{i}" for i in range(n)]
    return R(
        f"pa_om_chunk_scan_expand_{n}",
        Op.make("om_chunk_scan", *xs, dim="D"),
        Op.make("concat", *xs, dim="D"),
        law="A scheduled chunk-scan block denotes the concat of its "
        "chunk sums — the chunked decayed-scan schedule (FLA's "
        "batched chunk loop) over the same arms.",
        tags=_PA_FUS,
    )


PA_CHUNK_SCAN_EXPAND: list[Rewrite] = [
    _chunk_scan_expand(n) for n in range(2, 7)
]


# ---------------------------------------------------------------------------
#  §2 — Halide schedule primitives: tiling distributes
#
#  A reduction over a concat of tiles is the sum of the per-tile
#  reductions (Halide's split rdom / Triton's block-pointer chunked
#  reduce); a pointwise op computes inside the tile (``compute_at``
#  — the producer inlined at the consumer's tile granularity).
# ---------------------------------------------------------------------------

#: The concat axis must be exactly the reduced axis — an int RD or a
#: singleton tuple unwrapped.
_COND_CAT_AXIS = (
    "and",
    ("attr-cmp", ("attr", "CD"), "==", ("attr0", "RD")),
    (
        "or",
        ("attr-type", "RD", "int"),
        (
            "and",
            ("attr-type", "RD", "tuple"),
            ("attr-len", "RD", "==", 1),
        ),
    ),
)

pa_sum_over_concat = R(
    "pa_sum_over_concat",
    Op.make(
        "sum",
        Op.make("concat", "a", "b", dim="CD"),
        dim="RD",
        keepdim="K",
    ),
    Op.make(
        "add",
        Op.make("sum", "a", dim="RD", keepdim="K"),
        Op.make("sum", "b", dim="RD", keepdim="K"),
    ),
    cond=_COND_CAT_AXIS,
    law="Σ over a concat boundary = Σ per tile added — reduction "
    "tiling (Halide's split rdom; the partial-sum decomposition "
    "every reduce kernel performs).",
    tags=_PA_CAT,
)


def _split_pull(op: str) -> Rewrite:
    """``split(f(u)) = f(split(u))`` — the pointwise op tiles inside.

    Halide's ``compute_at``: the producer is inlined at the
    consumer's tile granularity, so slicing then evaluating equals
    evaluating then slicing — for a pure elementwise ``f`` any
    ``sizes``/``dim``/``index`` tile spec works.
    """
    return R(
        f"pa_split_pull_{op}",
        Op.make(
            "split",
            Op.make(op, "u"),
            sizes="SZ",
            dim="D",
            index="I",
        ),
        Op.make(
            op,
            Op.make("split", "u", sizes="SZ", dim="D", index="I"),
        ),
        law=f"split({op}(u)) = {op}(split(u)) — the elementwise op "
        "commutes with the tile slice (Halide compute_at: compute "
        "the producer inside the consumer's tile).",
        tags=_PA_SIM,
    )


PA_SPLIT_PULL: list[Rewrite] = [
    _split_pull(op)
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
]


# ---------------------------------------------------------------------------
#  §3 — GEMM operand folds (cuBLAS alpha, F.linear's weight prepack)
# ---------------------------------------------------------------------------

pa_mm_scale_alpha_w = R(
    "pa_mm_scale_alpha_w",
    Op.make("mul", Op.make("matmul", "a", "w"), "s"),
    Op.make("matmul", "a", Op.make("mul", "w", "s")),
    law="(a@w)·s = a@(w·s) for scalar s — the alpha scale folds "
    "into the weight operand (cuBLAS's alpha; the quant/LoRA "
    "weight-side scale migration).",
    check=_scalar,
    tags=_PA_CAT,
)

pa_mm_scale_alpha_x = R(
    "pa_mm_scale_alpha_x",
    Op.make("mul", Op.make("matmul", "a", "w"), "s"),
    Op.make("matmul", Op.make("mul", "a", "s"), "w"),
    law="(a@w)·s = (a·s)@w for scalar s — alpha on the activation "
    "side instead (the same contraction-scale naturality).",
    check=_scalar,
    tags=_PA_CAT,
)

#: F.linear(x, W) denotes x @ Wᵀ — the substrate's mm-shape guard
#: with W's last-two swap (``transpose-out`` spec).
_COND_MM_LINEAR = (
    "and",
    ("rank", "W", "==", 2),
    ("mm-shape-ok", "x", ("transpose-out", "W", -2, -1)),
)

pa_mm_to_linear = R(
    "pa_mm_to_linear",
    Op.make("matmul", "x", Op.make("transpose", "W", dim0=-2, dim1=-1)),
    Op.make("linear", "x", "W"),
    law="x @ Wᵀ = linear(x, W) — the F.linear slot IS a GEMM with "
    "the weight pre-transposed (the prepack cuBLASLt/oneDNN do "
    "once, not per tile).",
    cond=_COND_MM_LINEAR,
    tags=_PA_SIM,
)

pa_linear_expand = R(
    "pa_linear_expand",
    Op.make("linear", "x", "W"),
    Op.make("matmul", "x", Op.make("transpose", "W", dim0=-2, dim1=-1)),
    law="linear(x, W) denotes x @ Wᵀ — the library op's spelled "
    "contraction.",
    cond=_COND_MM_LINEAR,
    tags=_PA_SIM,
)


# ---------------------------------------------------------------------------
#  §4 — Strength reductions and constant folding on kernels
#
#  The InstCombine/instruction-selection class: a device kernel
#  emits the cheap elementary op for the expensive one the program
#  spelled.  These are one-directional canonicalizations — safe in
#  the bounded e-graph (nothing folds back).
# ---------------------------------------------------------------------------

_COND_CONST_NONZERO = (
    "and",
    ("const-num", "C"),
    ("not", ("const-cmp", "C", "==", 0)),
)

pa_div_const_recip = R(
    "pa_div_const_recip",
    Op.make("div", "u", "C"),
    Op.make("mul", "u", Op.make("reciprocal", "C")),
    law="u / c = u · recip(c) for a constant divisor — InstCombine's "
    "div-by-const strength reduction (the kernel emits the "
    "multiply; the const itself folds in a second step).",
    cond=_COND_CONST_NONZERO,
    tags=_PA_SIM,
)

pa_const_fold_recip = R(
    "pa_const_fold_recip",
    Op.make("reciprocal", "C"),
    "R",
    law="recip(c) = 1/c for a numeric literal — constant folding "
    "at compile time (the recip lands in the folded constant).",
    cond=_COND_CONST_NONZERO,
    dspec={"$leaf:R": ("recip", ("const", "C"))},
    tags=_PA_SIM,
)

pa_const_fold_neg = R(
    "pa_const_fold_neg",
    Op.make("neg", "C"),
    "R",
    law="neg(c) = −c for a numeric literal — constant folding.",
    cond=("const-num", "C"),
    dspec={"$leaf:R": ("neg", ("const", "C"))},
    tags=_PA_SIM,
)

pa_exp_to_exp2 = R(
    "pa_exp_to_exp2",
    Op.make("exp", "x"),
    Op.make("exp2", Op.make("mul", "x", Const(1.4426950408889634))),
    law="exp(x) = exp2(x·log2 e) — the device-expansion strength "
    "reduction: hardware exp IS ex2 (CUDA ``expf`` lowers to "
    "``ex2.approx`` scaled by log2(e)).",
    tags=_PA_SIM,
)

pa_log_to_log2 = R(
    "pa_log_to_log2",
    Op.make("log", "x"),
    Op.make("mul", Op.make("log2", "x"), Const(0.6931471805599453)),
    law="log(x) = log2(x)·ln 2 — the same instruction-selection "
    "canonicalization one family over.",
    tags=_PA_SIM,
)

pa_pow2_exp2 = R(
    "pa_pow2_exp2",
    Op.make("pow", "C", "x"),
    Op.make("exp2", "x"),
    law="pow(2, x) = exp2(x) — the base-2 power IS the hardware "
    "exponential (no log/exp round-trip needed).",
    cond=("const-cmp", "C", "==", 2),
    tags=_PA_SIM,
)

pa_sigmoid_tanh = R(
    "pa_sigmoid_tanh",
    Op.make("sigmoid", "x"),
    Op.make(
        "mul",
        Const(0.5),
        Op.make(
            "add",
            Op.make("tanh", Op.make("mul", "x", Const(0.5))),
            Const(1),
        ),
    ),
    law="σ(x) = ½·(tanh(x/2) + 1) — the tanh-spelled sigmoid XLA "
    "and the fast-math paths emit where σ is absent.",
    tags=_PA_SIM,
)

pa_rsqrt_expand = R(
    "pa_rsqrt_expand",
    Op.make("rsqrt", "u"),
    Op.make("reciprocal", Op.make("sqrt", "u")),
    law="rsqrt(u) = 1/√u — the device instruction's spelled "
    "denotation (the reciprocal root every GPU ISA carries).",
    tags=_PA_SIM,
)

pa_mean_expand = R(
    "pa_mean_expand",
    Op.make("mean", "u", dim="D", keepdim="K"),
    Op.make("mul", Op.make("sum", "u", dim="D", keepdim="K"), "N"),
    law="mean(u, d) = sum(u, d)·(1/N) — a mean kernel is the "
    "reduce plus a scale by the reduced count (the codegen "
    "lowering, N folded to a constant).",
    check=_mean_guard,
    derive=_mean_count,
    tags=_PA_SIM,
)


# ---------------------------------------------------------------------------
#  §5 — Layout no-ops: views/copies erase at codegen
# ---------------------------------------------------------------------------

pa_contiguous_id = R(
    "pa_contiguous_id",
    Op.make("contiguous", "u"),
    "u",
    law="contiguous(u) = u at the value level — the copy is a "
    "layout decision, not a function (Inductor's layout strip).",
    tags=_PA_SIM,
)

pa_detach_id = R(
    "pa_detach_id",
    Op.make("detach", "u"),
    "u",
    law="detach(u) = u — autograd metadata erases at codegen.",
    tags=_PA_SIM,
)

pa_detach_inplace_id = R(
    "pa_detach_inplace_id",
    Op.make("detach_", "u"),
    "u",
    law="detach_(u) = u — the in-place variant erases identically.",
    tags=_PA_SIM,
)


# ---------------------------------------------------------------------------
#  The bank
# ---------------------------------------------------------------------------

KERNEL_PRIOR_ART_RULES: tuple[Rewrite, ...] = (
    # -- fused-kernel denotations (the compiled composite IS the
    #    spelled body)
    pa_kf_silu_expand,
    pa_kf_softsign_expand,
    pa_kf_addmul_expand,
    pa_kf_swiglu_expand,
    pa_kf_silu_sigmoid_expand,
    pa_kf_addmul_tanh_expand,
    pa_kf_addmul2_expand,
    pa_kf_mul3_expand,
    pa_kf_softmax_expand,
    pa_kf_softmax_mm_expand,
    pa_kf_sdpa_expand,
    pa_kf_sdpa_cap_expand,
    pa_kf_layernorm_expand,
    pa_kf_rmsnorm_expand,
    pa_kf_mm_bias_expand,
    pa_kf_mm_scale_expand,
    pa_kf_mm_sb_expand,
    pa_kf_mm_gelu_expand,
    pa_om_decayed_expand,
    *PA_CHUNK_SCAN_EXPAND,
    # -- Halide schedule primitives (tiling distributions)
    pa_sum_over_concat,
    *PA_SPLIT_PULL,
    # -- GEMM operand folds
    pa_mm_scale_alpha_w,
    pa_mm_scale_alpha_x,
    pa_mm_to_linear,
    pa_linear_expand,
    # -- strength reductions / constant folding
    pa_div_const_recip,
    pa_const_fold_recip,
    pa_const_fold_neg,
    pa_exp_to_exp2,
    pa_log_to_log2,
    pa_pow2_exp2,
    pa_sigmoid_tanh,
    pa_rsqrt_expand,
    pa_mean_expand,
    # -- layout no-ops
    pa_contiguous_id,
    pa_detach_id,
    pa_detach_inplace_id,
)
