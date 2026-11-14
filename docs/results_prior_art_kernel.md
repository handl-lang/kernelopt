# Kernel prior-art bank — results

> Snapshot note: this run predates `pa_kf_sdpa_mask_expand` /
> `pa_kf_sdpa_gqa_expand` — the bank is 51 laws today (49 here).

The kernel-level prior-art encoding the novelty sieve's derivability floor was missing: ``mlopt_kernel.priorart.KERNEL_PRIOR_ART_RULES`` — 49 laws, all tagged ``PRIOR_ART``, opt-in through ``known_bank()`` (never in ``KERNEL_RULES`` — they are premises, not play actions).  Where the substrate's ``PRIOR_ART`` encodes the *tensor-level* literature (TASO substitutions, Inductor/XLA rewrites), this bank encodes the **kernel codegen vocabulary** — what Halide/Triton/TVM/cuBLAS already do to a spelled program:

- **Fused-kernel denotations (24)** — each ``kf_*``/  ``om_*`` op expands to its spelled composite (the sink's   own binding body): SiLU/softsign/FMA/SwiGLU gates, the   residue folds, the spelled exp/Σexp softmax site (the   Triton-tutorial fused kernel), softmax·mm, whole-block   SDPA (FlashAttention)   plain and soft-capped (Gemma-2), layernorm/rmsnorm   (Apex/Triton), the cuBLASLt epilogues   (``addmm``/scale/scale-bias/GELU), the decayed-score   ``om_decayed`` element and the chunked-scan   ``om_chunk_scan`` blocks (FLA's chunked schedule —   arities 2–6).
- **Halide schedule primitives (9)** — a ``sum`` over a   concat of tiles is the per-tile sums added (split rdom);   a pointwise op computes inside the tile (``compute_at``   — ``split`` pull-throughs for neg/abs/relu/sigmoid/silu/  tanh/exp/square).
- **GEMM operand folds (4)** — alpha scale into weight or   activation (scalar-guarded), and the ``linear``↔  ``matmul(x, Wᵀ)`` prepack pair.
- **Strength reductions / const folds (9)** — div-by-const   → mul-by-recip (two laws, InstCombine), ``neg(c)`` fold,   ``exp``→``exp2·log2e``, ``log``→``log2·ln2``,   ``pow(2,x)``→``exp2(x)``, the tanh-spelled sigmoid,   ``rsqrt``→``recip(sqrt)`` denotation, ``mean``→  ``sum·(1/N)``.
- **Layout no-ops (3)** — ``contiguous``/``detach``/  ``detach_`` erase at codegen.

**Verification** — the kernel port of the parent's ``prior_art_verify``: each rule's accepted sites (curated ``CASES`` for the enum-unreachable corners, then the ``oracle._binding_envs`` sweep) evaluated fp64 through ``KernelSink.verify``; the first ``equal`` site must carry a standalone-replayable ``verify_law`` certificate.

**Result: 49/49 certified** — verdicts: certified=49.

## Sieve impact — the floor before and after

``known_bank()`` grew 264 → 313 rules.  The previous sweep (tensor-catalog floor only) reported ``known=1, review=20, unmeasured=4, unpaid=1``; with the kernel vocabulary encoded the same 26 candidates report **``known=26``** — every ``kern_fuse_*`` fold is derivable through its fused op's denotation bridge (the flash/apex/cuBLAS-epilogue/fma vocabulary is known codegen, measured), the five ``kern_fuse_chunk_scan_*`` members derive through the scheduled-block denotation (FLA's chunked scan), and ``kern_div_recip`` composes ``pa_div_const_recip`` + ``pa_const_fold_recip`` (InstCombine's div-by-const plus constant folding).  The review queue is **empty**: what the old run called underivable meant "absent from our tensor catalog", not "unknown to kernel compilers".

| law | before | after | witness |
|---|---|---|---|
| `kern_fuse_silu` … `kern_fuse_om_decayed` (19 fused folds) | review/unpaid | **known** | `pa_kf_*_expand` / `pa_om_decayed_expand` |
| `kern_fuse_chunk_scan` + `_3`…`_6` (5 schedule folds) | review/unmeasured | **known** | `pa_om_chunk_scan_expand_{2..6}` |
| `kern_div_recip` | review | **known** | `pa_div_const_recip` + `pa_const_fold_recip` |
| `kern_square_strength` | known | **known** | `pow_to_square` + `square_expand` (unchanged) |

| law | verdict | accepted | equal | unequal | skip | cert |
|---|---|---|---|---|---|---|
| `pa_kf_silu_expand` | **certified** | 24 | 19 | 0 | 5 | yes |
| `pa_kf_softsign_expand` | **certified** | 24 | 17 | 0 | 7 | yes |
| `pa_kf_addmul_expand` | **certified** | 24 | 23 | 0 | 1 | yes |
| `pa_kf_swiglu_expand` | **certified** | 24 | 18 | 0 | 6 | yes |
| `pa_kf_silu_sigmoid_expand` | **certified** | 24 | 14 | 0 | 10 | yes |
| `pa_kf_addmul_tanh_expand` | **certified** | 24 | 17 | 0 | 7 | yes |
| `pa_kf_addmul2_expand` | **certified** | 24 | 24 | 0 | 0 | yes |
| `pa_kf_mul3_expand` | **certified** | 24 | 23 | 0 | 1 | yes |
| `pa_kf_softmax_expand` | **certified** | 24 | 12 | 0 | 12 | yes |
| `pa_kf_softmax_mm_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_sdpa_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_sdpa_cap_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_layernorm_expand` | **certified** | 24 | 6 | 0 | 18 | yes |
| `pa_kf_rmsnorm_expand` | **certified** | 24 | 6 | 0 | 18 | yes |
| `pa_kf_mm_bias_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_mm_scale_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_mm_sb_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_kf_mm_gelu_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_decayed_expand` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_chunk_scan_expand_2` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_chunk_scan_expand_3` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_chunk_scan_expand_4` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_chunk_scan_expand_5` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_om_chunk_scan_expand_6` | **certified** | 24 | 1 | 0 | 23 | yes |
| `pa_sum_over_concat` | **certified** | 24 | 2 | 0 | 22 | yes |
| `pa_split_pull_neg` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_abs` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_relu` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_sigmoid` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_silu` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_tanh` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_exp` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_split_pull_square` | **certified** | 12 | 12 | 0 | 0 | yes |
| `pa_mm_scale_alpha_w` | **certified** | 24 | 2 | 0 | 22 | yes |
| `pa_mm_scale_alpha_x` | **certified** | 24 | 2 | 0 | 22 | yes |
| `pa_mm_to_linear` | **certified** | 17 | 17 | 0 | 0 | yes |
| `pa_linear_expand` | **certified** | 17 | 17 | 0 | 0 | yes |
| `pa_div_const_recip` | **certified** | 24 | 24 | 0 | 0 | yes |
| `pa_const_fold_recip` | **certified** | 18 | 18 | 0 | 0 | yes |
| `pa_const_fold_neg` | **certified** | 18 | 18 | 0 | 0 | yes |
| `pa_exp_to_exp2` | **certified** | 17 | 16 | 0 | 1 | yes |
| `pa_log_to_log2` | **certified** | 17 | 5 | 0 | 1 | yes |
| `pa_pow2_exp2` | **certified** | 21 | 17 | 0 | 4 | yes |
| `pa_sigmoid_tanh` | **certified** | 17 | 16 | 0 | 1 | yes |
| `pa_rsqrt_expand` | **certified** | 17 | 4 | 0 | 1 | yes |
| `pa_mean_expand` | **certified** | 24 | 16 | 0 | 8 | yes |
| `pa_contiguous_id` | **certified** | 17 | 16 | 0 | 1 | yes |
| `pa_detach_id` | **certified** | 17 | 16 | 0 | 1 | yes |
| `pa_detach_inplace_id` | **certified** | 17 | 16 | 0 | 1 | yes |

## Dropped / failed laws

None — every encoded law certified on at least one fp64-equal accepted site.

## Honest misses

- *Vectorize/unroll/pipeline/double-buffer* — no term-level expression: the IR has no lane or stage axis; these are loop-schedule attributes, not equivalences a rewrite can state.
- *The fold direction of every `kern_fuse_*` law* — alpha-siblings of the shipped bank; the sieve's premise cut excludes them, so encoding them would be circular.  The expand direction is what prior art honestly claims.
- *Online/max-shifted softmax* — no reduce-max op in the IR, so FlashAttention's running-max form cannot be spelled; the value-level `softmax` denotation stands in.
- `mul(x,0)=0` / `div(x,x)=1` class — IEEE-unsound at NaN/Inf, same miss as the tensor bank's.
