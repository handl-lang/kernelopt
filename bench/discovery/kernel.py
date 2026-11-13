"""The kernelopt spike's bench cases — measured codegen pay.

Forty-six kernel cases: a spelled program, a codegen-law candidate, and
the honest row — lower both through
:class:`mlopt_kernel.KernelSink`, verify numerically (verify-first),
then time spelled vs scheduled.  Where the case has a registered
hand-Triton kernel (``bench.limits.triton``), the row times and
verifies that arm too — the practitioner's kernel is the field
baseline the fused composite is measured against.  This is the
spike's measured half of the go/no-go: does the engine emit a
verified-equal kernel at lower measured cost than the spelled
baseline.

Run::

    .venv/bin/python -m bench.discovery.kernel
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, TensorType, Var

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost

#: (name, category, spelled builder, rules to fire in order)
CASES: tuple[tuple[str, str, Any, tuple[str, ...]], ...] = ()


def _v(name, shape):
    return Var(name, TensorType(tuple(shape)))


def _mk():
    x = _v("x", (4096, 512))
    y = _v("y", (4096, 512))
    w = _v("w", (512, 512))
    b = _v("b", (512,))
    return x, y, w, b


def _mk_decayed():
    s = _v("s", (16, 128, 128))
    d = _v("d", (128, 128))
    v = _v("v", (16, 128, 128))
    return s, d, v


def _mk_decayed_cases():
    """The carrier-style decayed-scan case builders — real spellings.

    The ``om_decayed`` vocabulary measured against the spellings the
    chunked decayed scan actually emits: the generated ``decay`` mask,
    the ``linear``-produced score, the ``decay_vec``-gated cross term,
    and the ``split``-chunked pair.  Every decay rate is a live
    ``sigmoid(g)`` feed — the carrier parametrisation (g a learned
    scalar, not a baked literal: a ``Const`` rate would pin the mask
    generator to CPU and break under CUDA feeds).
    """
    # the generated-mask element: (s * tril(g^(i-j))) @ v
    sm = _v("sm", (16, 128, 128))
    gm = _v("gm", ())
    vm = _v("vm", (16, 128, 128))
    mask = Op.make("decay", Op.make("sigmoid", gm), t=128)
    # the linear-projected score: (linear(x,w) ⊙ decay) @ v
    xi = _v("xi", (128, 64))
    wi = _v("wi", (128, 64))
    gi = _v("gi", ())
    vi = _v("vi", (128, 64))
    # the two-term block arm: intra element + decay_vec cross term
    sp = _v("sp", (128, 128))
    gp = _v("gp", ())
    vp = _v("vp", (128, 64))
    gr = _v("gr", ())
    xp = _v("xp", (128, 64))
    wp = _v("wp", (64, 64))
    up = _v("up", (64, 64))
    row_gate = Op.make(
        "unsqueeze",
        Op.make("decay_vec", Op.make("sigmoid", gr), t=128),
        dim=-1,
    )
    # the two-chunk scan: split views, per-chunk om_decayed, summed
    sc = _v("sc", (16, 128, 64))
    gc = _v("gc", ())
    vc = _v("vc", (16, 128, 64))
    chunk_decay = Op.make("decay", Op.make("sigmoid", gc), t=64)

    def chunk(i):
        return Op.make(
            "matmul",
            Op.make(
                "mul",
                Op.make("split", sc, sizes=(64, 64), dim=1, index=i),
                chunk_decay,
            ),
            Op.make("split", vc, sizes=(64, 64), dim=1, index=i),
        )

    # the composed decayed variants — a double-masked element, the
    # arm's div-peel, and the flat 3-ary concat.
    dm1 = _v("dm1", (128, 128))
    dm2 = _v("dm2", (128, 128))
    sd = _v("sd", (16, 128, 128))
    dm = _v("dm", (128, 128))
    vd = _v("vd", (16, 128, 128))
    # the three-chunk scan: same split/decay_vec spelling as the
    # binary carrier case, one more arm — sc3/vc3 carry 3·64 rows.
    sc3 = _v("sc3", (16, 192, 64))
    vc3 = _v("vc3", (16, 192, 64))
    gc3 = _v("gc3", ())
    chunk_decay3 = Op.make("decay", Op.make("sigmoid", gc3), t=64)
    row_gate3 = Op.make(
        "unsqueeze",
        Op.make("decay_vec", Op.make("sigmoid", gc3), t=64),
        dim=-1,
    )
    split3 = (64, 64, 64)

    def chunk3_intra(i):
        return Op.make(
            "matmul",
            Op.make(
                "mul",
                Op.make("split", sc3, sizes=split3, dim=1, index=i),
                chunk_decay3,
            ),
            Op.make("split", vc3, sizes=split3, dim=1, index=i),
        )

    def chunk3_cross(i, j):
        return Op.make(
            "mul",
            row_gate3,
            Op.make(
                "div",
                Op.make(
                    "matmul",
                    Op.make("split", sc3, sizes=split3, dim=1, index=i),
                    Op.make(
                        "matmul",
                        Op.make(
                            "split",
                            sc3,
                            sizes=split3,
                            dim=1,
                            index=j,
                        ),
                        Op.make(
                            "split",
                            vc3,
                            sizes=split3,
                            dim=1,
                            index=j,
                        ),
                    ),
                ),
                Const(2),
            ),
        )

    def arm3(i):
        t = chunk3_intra(i)
        for j in range(i):
            t = Op.make("add", t, chunk3_cross(i, j))
        return t

    return (
        (
            "kernel_decayed_mask",
            "speed",
            Op.make("matmul", Op.make("mul", sm, mask), vm),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_intra",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "mul",
                    Op.make("linear", xi, wi),
                    Op.make("decay", Op.make("sigmoid", gi), t=128),
                ),
                vi,
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_pair",
            "speed",
            Op.make(
                "add",
                Op.make(
                    "matmul",
                    Op.make(
                        "mul",
                        sp,
                        Op.make("decay", Op.make("sigmoid", gp), t=128),
                    ),
                    vp,
                ),
                Op.make(
                    "mul",
                    row_gate,
                    Op.make("matmul", xp, Op.make("matmul", wp, up)),
                ),
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_chunks",
            "speed",
            Op.make("add", chunk(0), chunk(1)),
            ("kern_fuse_om_decayed",),
        ),
        # The chunked-scan schedule — a ``concat`` of two chunk arms
        # (intra element per arm + a decay-gated cross pair product on
        # arm 1), the retnet carrier's extraction shape.  The spelled
        # tape launches every piece per chunk; ``kern_fuse_chunk_scan``
        # folds the concat to one scheduled ``om_chunk_scan`` block.
        (
            "kernel_chunk_scan",
            "speed",
            Op.make(
                "concat",
                Op.make(
                    "matmul",
                    Op.make(
                        "mul",
                        Op.make(
                            "split",
                            sc,
                            sizes=(64, 64),
                            dim=1,
                            index=0,
                        ),
                        chunk_decay,
                    ),
                    Op.make(
                        "split", vc, sizes=(64, 64), dim=1, index=0
                    ),
                ),
                Op.make(
                    "add",
                    Op.make(
                        "matmul",
                        Op.make(
                            "mul",
                            Op.make(
                                "split",
                                sc,
                                sizes=(64, 64),
                                dim=1,
                                index=1,
                            ),
                            chunk_decay,
                        ),
                        Op.make(
                            "split",
                            vc,
                            sizes=(64, 64),
                            dim=1,
                            index=1,
                        ),
                    ),
                    Op.make(
                        "mul",
                        Op.make(
                            "unsqueeze",
                            Op.make(
                                "decay_vec",
                                Op.make("sigmoid", gc),
                                t=64,
                            ),
                            dim=-1,
                        ),
                        Op.make(
                            "div",
                            Op.make(
                                "matmul",
                                Op.make(
                                    "split",
                                    sc,
                                    sizes=(64, 64),
                                    dim=1,
                                    index=1,
                                ),
                                Op.make(
                                    "matmul",
                                    Op.make(
                                        "split",
                                        sc,
                                        sizes=(64, 64),
                                        dim=1,
                                        index=0,
                                    ),
                                    Op.make(
                                        "split",
                                        vc,
                                        sizes=(64, 64),
                                        dim=1,
                                        index=0,
                                    ),
                                ),
                            ),
                            Const(2),
                        ),
                    ),
                ),
                dim=-2,
            ),
            ("kern_fuse_chunk_scan",),
        ),
        # The masked+decayed combo — ``(s ⊙ d₁ ⊙ d₂) @ v``: the
        # element's mask is itself a product (decay ⊙ causal), so
        # ``kern_fuse_mul3`` and ``kern_fuse_om_decayed`` compete
        # over the same site — the composition surface a single
        # decayed element lacks.
        (
            "kernel_decayed_mask2",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "mul",
                    Op.make("mul", sm, dm1),
                    dm2,
                ),
                vm,
            ),
            ("kern_fuse_mul3", "kern_fuse_om_decayed"),
        ),
        # The scan arm's own scale peel — ``(s ⊙ d) @ v / S``: the
        # intra element under the constant divisor the chunked
        # spelling divides every block by; the recip fold chains
        # out of the decayed fold.
        (
            "kernel_decayed_div",
            "speed",
            Op.make(
                "div",
                Op.make("matmul", Op.make("mul", sd, dm), vd),
                Const(2.0),
            ),
            ("kern_fuse_om_decayed", "kern_div_recip"),
        ),
        # The flat 3-ary scan — the same concat-of-chunks site as
        # ``kernel_chunk_scan`` but spelled flat, the arity the
        # ``kern_fuse_chunk_scan_3`` family member owns (the
        # registered corpus had only the nested-binary spelling).
        (
            "kernel_chunk_scan3",
            "speed",
            Op.make("concat", arm3(0), arm3(1), arm3(2), dim=-2),
            ("kern_fuse_chunk_scan_3",),
        ),
    )


def _mk_decayed_scale_cases():
    """The ``om_decayed`` size ladder — where the fold stops paying.

    Same ``matmul(s⊙d, v)`` site as ``kernel_decayed``, T swept
    512→4096 (B sized down to bound memory: the fold's win is one
    less ``(…,T,K)`` buffer, the premium is the per-op compile
    dispatch — the earlier scale probe measured the ratio climbing
    0.80→0.88→0.99 and never inverting; this ladder prices the
    crossover directly on the timed row).
    """
    out = []
    for t, n in ((512, 8), (1024, 4), (2048, 2), (4096, 1)):
        s = _v(f"sl{t}", (n, t, t))
        d = _v(f"dl{t}", (t, t))
        v = _v(f"vl{t}", (n, t, t))
        out.append(
            (
                f"kernel_decayed_t{t}",
                "speed",
                Op.make("matmul", Op.make("mul", s, d), v),
                ("kern_fuse_om_decayed",),
            )
        )
    return tuple(out)


def _mk_chain_cases():
    """The multi-op stratum — composed spellings, not single sites.

    The overnight loop's thin mint (10 episodes, 3 minted on the
    single-op head of the registry) was a corpus-bound artefact: one
    fusion site per term leaves ``synthesize_rules`` no pair surface
    and the census mint almost nothing to bind.  These cases spell
    the chains real kernels actually carry — norm+activation
    stacks, attention interiors, mm epilogue stacks, and the
    workload spellings (rotary, gated residual, grouped dequant,
    kv-cache concat) the heavy modules lower to.  ``--episodes N``
    reads the registry in order, so the composed stratum leads:
    the first-N slice now reaches terms whose subterm trees the
    mint can compose over.
    """
    x = _v("x", (4096, 512))
    y = _v("y", (4096, 512))
    w = _v("w", (512, 512))
    b = _v("b", (512,))
    m = _v("m", (4096, 512))
    g = _v("g", (4096, 512))
    cv = _v("cv", (4096, 512))
    sv = _v("sv", (4096, 512))
    kn = _v("kn", (256, 512))
    kc = _v("kc", (256, 512))
    wq = _v("wq", (512, 512))
    zp = _v("zp", (512,))
    sc = _v("sc", (512,))

    # the spelled norm blocks — the same sites the single-op cases
    # carry, here as *operands* of the composed terms.
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    layernorm = Op.make(
        "mul",
        ln_center,
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", ln_center, Const(2)),
                    dim=-1,
                    keepdim=True,
                ),
                Const(1e-5),
            ),
        ),
    )
    rmsnorm = Op.make(
        "mul",
        x,
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", x, Const(2)),
                    dim=-1,
                    keepdim=True,
                ),
                Const(1e-6),
            ),
        ),
    )

    def softmax_of(u):
        return Op.make(
            "div",
            Op.make("exp", u),
            Op.make("sum", Op.make("exp", u), dim=-1, keepdim=True),
        )

    # rotary's rotate-half: cat(-x_hi, x_lo) over the last axis —
    # the split/neg/concat view chain the rope kernels spell.
    lo = Op.make("split", x, sizes=(256, 256), dim=-1, index=0)
    hi = Op.make("split", x, sizes=(256, 256), dim=-1, index=1)
    rot = Op.make("concat", Op.make("neg", hi), lo, dim=-1)

    return (
        # The pre-norm residual block — ``x + gelu(ln(x))``: the
        # norm block spelled inside the pointwise tail, not alone.
        (
            "kernel_ln_gelu_res",
            "speed",
            Op.make("add", Op.make("gelu", layernorm), x),
            ("kern_fuse_layernorm",),
        ),
        # The gated-MLP norm chain — ``swiglu(rms(x), y)``: the
        # swiglu site binds with the norm block as its gate child.
        (
            "kernel_rms_swiglu",
            "speed",
            Op.make(
                "mul",
                Op.make("mul", rmsnorm, Op.make("sigmoid", rmsnorm)),
                y,
            ),
            ("kern_fuse_rmsnorm", "kern_fuse_silu", "kern_fuse_swiglu"),
        ),
        # The pre-norm MLP — ``gelu(ln(x) @ w)``: the mm epilogue
        # fold's activation site sits over a norm operand.
        (
            "kernel_ln_mm_gelu",
            "speed",
            Op.make("gelu", Op.make("matmul", layernorm, w)),
            ("kern_fuse_layernorm", "kern_fuse_mm_gelu"),
        ),
        # The full mm epilogue stack — ``gelu(mm·s + b)``: scale,
        # bias and activation over the GEMM, the spelling the
        # kf_mm_sb fold absorbs mid-chain.
        (
            "kernel_mm_sb_gelu",
            "speed",
            Op.make(
                "gelu",
                Op.make(
                    "add",
                    Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                    b,
                ),
            ),
            ("kern_fuse_mm_scale_bias", "kern_fuse_addmul"),
        ),
        # The attention block with its residual — ``x + softmax(qk·s)
        # @ v``: the whole-sdpa site inside a residual add.
        (
            "kernel_attn_residual",
            "speed",
            Op.make(
                "add",
                x,
                Op.make(
                    "matmul",
                    softmax_of(
                        Op.make(
                            "mul",
                            Op.make("matmul", x, w),
                            Const(0.25),
                        )
                    ),
                    w,
                ),
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa",
            ),
        ),
        # The masked-softmax interior — ``softmax(qk·s + m)``: the
        # additive-mask spelling the attention kernels fold where a
        # dropout node would sit eval-side (the sink has no dropout
        # op — the mask add is the live composition surface).
        (
            "kernel_softmax_masked",
            "speed",
            softmax_of(
                Op.make(
                    "add",
                    Op.make(
                        "mul", Op.make("matmul", x, w), Const(0.25)
                    ),
                    m,
                )
            ),
            ("kern_fuse_mm_scale", "kern_fuse_softmax"),
        ),
        # The rotary pair — ``x·cos + rot(x)·sin``: two products over
        # a split/neg/concat view chain, the addmul2 site the
        # pointwise-pair fold owns.
        (
            "kernel_rope",
            "memory",
            Op.make(
                "add",
                Op.make("mul", x, cv),
                Op.make("mul", rot, sv),
            ),
            ("kern_fuse_addmul", "kern_fuse_addmul2"),
        ),
        # The kv-cache concat — ``(q @ cat(k_new, k_cache))·s``: the
        # score matmul's operand materialises the cache append; the
        # scale epilogue folds over the concat.
        (
            "kernel_kv_score",
            "speed",
            Op.make(
                "mul",
                Op.make("matmul", x, Op.make("concat", kn, kc, dim=-2)),
                Const(0.125),
            ),
            ("kern_fuse_mm_scale",),
        ),
        # The sigmoid-gated residual — ``x + y·sigmoid(g)``: the
        # highway gate shape; silu and addmul compete over the
        # product.
        (
            "kernel_gate_res",
            "speed",
            Op.make("add", Op.make("mul", y, Op.make("sigmoid", g)), x),
            ("kern_fuse_silu", "kern_fuse_addmul"),
        ),
        # The grouped dequant — ``x @ ((q-z)·s)``: the affine-weight
        # spelling inside the GEMM operand.  No bank fold reaches it
        # (``mul(sub, ·)`` is not a fusion site) — an honest flat row
        # whose subterm tree feeds the mint.
        (
            "kernel_dequant_mm",
            "speed",
            Op.make(
                "matmul",
                x,
                Op.make("mul", Op.make("sub", wq, zp), sc),
            ),
            (),
        ),
    )


def _cases():
    x, y, w, b = _mk()
    s, d, v = _mk_decayed()
    # the masked-sdpa arm: the additive mask is a live feed (causal
    # and padding masks materialise as operands); the gqa arm carries
    # 8 query heads over 4 kv heads — ``repeat(w|v, (2,1,1))``.
    mk = _v("mk", (4096, 512))
    qa = _v("qa", (8, 64, 64))
    kw = _v("kw", (4, 64, 64))
    kv = _v("kv", (4, 64, 64))
    gqa_rep = (2, 1, 1)
    kwr = Op.make("repeat", kw, repeats=gqa_rep)
    kvr = Op.make("repeat", kv, repeats=gqa_rep)
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    return (
        # The composed stratum leads the registry — the overnight
        # loop's ``--episodes N`` slice reads this order and the
        # single-op head gave the mint nothing to compose over.
        *_mk_chain_cases(),
        (
            "kernel_silu",
            "speed",
            Op.make("mul", x, Op.make("sigmoid", x)),
            ("kern_fuse_silu",),
        ),
        (
            "kernel_softsign",
            "speed",
            Op.make(
                "div", x, Op.make("add", Op.make("abs", x), Const(1))
            ),
            ("kern_fuse_softsign",),
        ),
        (
            "kernel_fma",
            "speed",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
            ("kern_fuse_addmul", "kern_fuse_addmul_tanh"),
        ),
        (
            "kernel_swiglu",
            "speed",
            Op.make("mul", Op.make("mul", x, Op.make("sigmoid", x)), y),
            ("kern_fuse_silu", "kern_fuse_swiglu"),
        ),
        (
            "kernel_softmax",
            "speed",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=-1, keepdim=True),
            ),
            ("kern_fuse_softmax",),
        ),
        (
            "kernel_layernorm",
            "speed",
            Op.make(
                "mul",
                ln_center,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", ln_center, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-5),
                    ),
                ),
            ),
            ("kern_fuse_layernorm",),
        ),
        (
            "kernel_mm_bias",
            "speed",
            Op.make("add", Op.make("matmul", x, w), b),
            ("kern_fuse_mm_bias",),
        ),
        (
            "kernel_mm_scale_bias",
            "speed",
            Op.make(
                "add",
                Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                b,
            ),
            ("kern_fuse_mm_scale_bias", "kern_fuse_addmul"),
        ),
        (
            "kernel_mm_scale",
            "speed",
            Op.make("mul", Op.make("matmul", x, w), Const(0.125)),
            ("kern_fuse_mm_scale",),
        ),
        (
            "kernel_mm_gelu",
            "speed",
            Op.make("gelu", Op.make("matmul", x, w)),
            ("kern_fuse_mm_gelu",),
        ),
        (
            "kernel_rmsnorm",
            "speed",
            Op.make(
                "mul",
                x,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", x, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-6),
                    ),
                ),
            ),
            ("kern_fuse_rmsnorm",),
        ),
        # The bounded-logit attention chain — ``softmax(tanh(qk))·v``,
        # the soft-cap spelling.  The un-capped ``softmax(qk)`` spelled
        # site overflows fp32 at bench scale (exp(qk) → inf where the
        # fused max-shifted softmax stays finite) — a documented
        # *stability* win for the fused op, kept out of the timed row
        # so the verify gate compares finite references.
        (
            "kernel_softmax_mm",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make("tanh", Op.make("matmul", x, w)),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make("tanh", Op.make("matmul", x, w)),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            ("kern_fuse_softmax", "kern_fuse_softmax_mm"),
        ),
        (
            "kernel_sdpa",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "mul",
                            Op.make("matmul", x, w),
                            Const(0.25),
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "mul",
                                Op.make("matmul", x, w),
                                Const(0.25),
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa",
            ),
        ),
        # The soft-capped twin — ``softmax(tanh(mm·s))·v``, the
        # Gemma-2 bounded-logit spelling.  ``kern_fuse_sdpa_cap``
        # folds the whole block to one ``kf_sdpa(cap=True)`` launch;
        # the chain laws compete as the two-launch fallback.
        (
            "kernel_sdpa_cap",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "tanh",
                            Op.make(
                                "mul",
                                Op.make("matmul", x, w),
                                Const(0.25),
                            ),
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "tanh",
                                Op.make(
                                    "mul",
                                    Op.make("matmul", x, w),
                                    Const(0.25),
                                ),
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa_cap",
            ),
        ),
        # The masked twin — ``softmax(mm·s + m)·v``: the
        # additive-mask spelling (causal masks materialise as the
        # operand).  ``kern_fuse_sdpa_mask`` folds the whole block to
        # one ``kf_sdpa(mask=True)`` launch; the chain laws compete
        # as the two-launch fallback (``m`` survives inside
        # ``kf_softmax_mm``'s score arg).
        (
            "kernel_sdpa_mask",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "add",
                            Op.make(
                                "mul",
                                Op.make("matmul", x, w),
                                Const(0.25),
                            ),
                            mk,
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "add",
                                Op.make(
                                    "mul",
                                    Op.make("matmul", x, w),
                                    Const(0.25),
                                ),
                                mk,
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa_mask",
            ),
        ),
        # The grouped-query twin — ``softmax(q·(k·g)·s)·(v·g)``: the
        # enable_gqa spelling, 8 query heads over 4 kv heads
        # materialised by ``repeat``.  ``kern_fuse_sdpa_gqa`` folds
        # the whole block to one ``kf_sdpa(gqa=(2,1,1))`` launch —
        # the repeats ride the attr instead of two spelled launches.
        (
            "kernel_sdpa_gqa",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "mul",
                            Op.make("matmul", qa, kwr),
                            Const(0.25),
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "mul",
                                Op.make("matmul", qa, kwr),
                                Const(0.25),
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                kvr,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa_gqa",
            ),
        ),
        (
            "kernel_decayed",
            "speed",
            Op.make(
                "matmul",
                Op.make("mul", s, d),
                v,
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_strength",
            "speed",
            Op.make(
                "add",
                Op.make("pow", x, Const(2)),
                Op.make("div", y, Const(4.0)),
            ),
            (
                "kern_square_strength",
                "kern_div_recip",
                "kern_fuse_addmul",
                "kern_fuse_addmul2",
            ),
        ),
        # The pointwise-pair residues — the two shapes the strength
        # reductions expose spelled.  ``kernel_powscale`` is the
        # weighted square ``pow(x,2)·y`` (→ ``mul(mul(x,x),y)`` →
        # ``kf_mul3``); ``kernel_addmul2`` is the two-product sum
        # ``x·y + y·b`` direct (→ ``kf_addmul2``).
        (
            "kernel_powscale",
            "speed",
            Op.make("mul", Op.make("pow", x, Const(2)), y),
            ("kern_square_strength", "kern_fuse_mul3"),
        ),
        (
            "kernel_addmul2",
            "speed",
            Op.make(
                "add",
                Op.make("mul", x, y),
                Op.make("mul", y, b),
            ),
            ("kern_fuse_addmul", "kern_fuse_addmul2"),
        ),
        # The reduction+strength pair — ``sqrt(Σx²)``: the L2 norm
        # spelling; the square fold reduces inside the reduce.
        (
            "kernel_l2norm",
            "memory",
            Op.make(
                "sqrt",
                Op.make(
                    "sum",
                    Op.make("pow", x, Const(2)),
                    dim=-1,
                    keepdim=True,
                ),
            ),
            ("kern_square_strength",),
        ),
        # The strength chain whole — ``x² · (y/4)``: both reductions
        # expose the left-nested product ``mul3`` folds — the
        # three-law composition ``kernel_strength`` splits across
        # two cases, spelled as one term.
        (
            "kernel_strength_mul",
            "speed",
            Op.make(
                "mul",
                Op.make("pow", x, Const(2)),
                Op.make("div", y, Const(4.0)),
            ),
            (
                "kern_square_strength",
                "kern_div_recip",
                "kern_fuse_mul3",
            ),
        ),
        (
            "kernel_mixed",
            "memory",
            Op.make(
                "mul",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("sigmoid", y),
            ),
            (
                "kern_fuse_silu",
                "kern_fuse_swiglu",
                "kern_fuse_silu_sigmoid",
            ),
        ),
        (
            "kernel_flat",
            "speed",
            Op.make("add", Op.make("tanh", x), Op.make("cos", y)),
            (),
        ),
        *_mk_decayed_cases(),
        *_mk_decayed_scale_cases(),
    )


@dataclass
class KernelRow:
    """One measured case row — spelled vs scheduled, verified."""

    name: str
    category: str
    scheduled: str = ""
    verified: bool | None = None
    max_abs: float | None = None
    cost_before: float = 0.0
    cost_after: float = 0.0
    spelled_us: float = 0.0
    scheduled_us: float = 0.0
    speedup: float | None = None
    triton_us: float | None = None
    triton_verified: bool | None = None
    triton_note: str = ""
    detail: dict = field(default_factory=dict)


def _time(fn, args, reps=50, device=None):
    import torch

    args = tuple(a.to(device) for a in args) if device else args
    for _ in range(10):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _vars_of(t):
    out = []

    def walk(u):
        if isinstance(u, Var):
            if u.name not in [v.name for v in out]:
                out.append(u)
        elif isinstance(u, Op):
            for a in u.args:
                walk(a)

    walk(t)
    return out


def _triton_arm(name):
    """Return ``(callable, None)`` for case *name*'s hand Triton kernel.

    The honest skip cases: triton not installed, no CUDA, or no
    kernel registered — the row records the reason, never a silent
    drop.
    """
    if importlib.util.find_spec("triton") is None:
        return None, "triton not installed"
    import torch

    if not torch.cuda.is_available():
        return None, "needs CUDA"
    from bench.limits.triton import kernel_builder

    mk = kernel_builder(name)
    if mk is None:
        return None, f"no triton kernel registered for {name!r}"
    return mk(), None


def run_case(
    name, category, term, rules_to_fire, *, device=None, reps=50
):
    """Board the spelled term, fire the laws, verify + time delivery."""
    import torch

    row = KernelRow(name=name, category=category)
    eg = EGraph()
    root = eg.add_term(term)
    for rn in rules_to_fire:
        r = next(r for r in KERNEL_RULES if r.name == rn)
        eg.apply_rule(r, root)
    best = eg.extract_best(root, kernel_cost) or term
    row.scheduled = repr(best)
    row.cost_before = kernel_cost(term)
    row.cost_after = kernel_cost(best)

    inputs = _vars_of(term)
    params = {}
    ir_s = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    )
    ir_b = IR(root=best, inputs=inputs, input_names=ir_s.input_names)
    sink = KernelSink()
    spelled = sink.lower(ir_s)
    sched = sink.lower(ir_b)
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float32)
        for v in inputs
    )
    rep = sink.verify(spelled, sched, feeds)
    row.verified = bool(rep.passed)
    row.max_abs = rep.max_abs
    row.spelled_us = _time(spelled, feeds, reps, device)
    row.scheduled_us = _time(sched, feeds, reps, device)
    if row.scheduled_us:
        row.speedup = round(row.spelled_us / row.scheduled_us, 3)

    arm, why = _triton_arm(name)
    if arm is None:
        row.triton_note = why or "unavailable"
    else:
        try:
            feeds_dev = tuple(a.to(device) for a in feeds)
            rep_t = sink.verify(spelled, arm, feeds_dev)
            row.triton_verified = bool(rep_t.passed)
            row.triton_us = _time(arm, feeds_dev, reps, device)
        except Exception as e:  # kernel failed — honest row
            row.triton_note = f"triton arm: {e}"
    return row


def main() -> list[KernelRow]:
    """Run every kernel case; print the honest row."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name, cat, term, fires in _cases():
        row = run_case(name, cat, term, fires, device=device)
        rows.append(row)
        tri = (
            f"triton {row.triton_us:.1f}us (verified={row.triton_verified})"
            if row.triton_us is not None
            else f"triton — ({row.triton_note})"
        )
        print(  # stdout-compat
            f"{name}: verified={row.verified} "
            f"cost {row.cost_before:.2f}->{row.cost_after:.2f} "
            f"{row.spelled_us:.1f}us -> {row.scheduled_us:.1f}us "
            f"({row.speedup}x), {tri} [{device}]"
        )
    return rows


if __name__ == "__main__":
    main()
