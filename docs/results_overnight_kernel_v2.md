# Kernel overnight discovery run

Episodes -> mint -> admit -> classify on the kernel corpus.  Each episode's arena pool is minted *from the case's own term*: the shared corpus-census proposals that bind it plus the ``syn_*`` pair-composites ``meta.synthesize_rules`` mints over ``KERNEL_RULES`` (+ the substrate's guarded parents) seeded on the term.  Every mint faces the emit gate (strict LHS bind + instantiated RHS must denote).  Evaluation binds a firing site, verifies the concrete pair through ``KernelSink`` (eager bodies, fp64) and prices the firing in ``kernel_cost``.  The admission gauntlet is the kernel referee — ``oracle`` (instance evaluates ``equal``), ``pay`` (the rule alone must extract a *verified* term strictly below the spelled cost on a registered case), ``sweep`` (all bound corpus sites equal), ``cert`` (the instance carries a standalone-replayable certificate).  Admitted rules are classified by the novelty sieve's gates — **review** (cert ∧ beat ∧ ¬derivable: the bank cannot prove it) is the machine-original class.

- date: 2026-10-23 04:03 UTC · git: `e111e3c` · torch 2.14.1+cu130 on cuda
- protocol: `{"budget": 24, "cases": null, "dry": false, "episodes": 44, "fuel": 256, "mint": "kernel-corpus-compose", "page": 24, "player": "playbook", "referee": "full", "rounds": 4, "seed": 0, "vocab": "hand"}`
- episodes: 44 · evaluated: 129 · minted: 117 (111 composites, 97 guarded) · admitted: 97 · wall: 180.6s
- arena verdicts: false=1, known=16, true=1, unverified=9, useful=102

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 12 |
| pay | 8 |
| **admitted** | 97 |

## Novelty

| verdict | count |
|---|---|
| known | 96 |
| unpaid | 1 |

## Machine-original finds (sieve `review`)

**None.** Every admitted mint was already a theorem of the known bank — or nothing survived the gauntlet.  The funnel above is the measurement: composed kernel rules are derivable by construction when both parents are in the bank, so the honest expectation is refusals at ``pay`` (a composite never beats the floor its own chain reaches) and ``known`` verdicts at classify.

## Candidates

| candidate | family | arena | admitted | reason | novelty |
|---|---|---|---|---|---|
| `grammar:sub_to_add_dup` | algebraic-grammar | known | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_swiglu__kern_fuse_rmsnorm_6@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_swiglu__rms_norm_fold_nogain_8@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_syn_kern_fuse_silu__kern_fuse_rmsnorm_0@kernel_rms_swiglu__kern_fuse_rmsnorm_0@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_silu__kern_fuse_rmsnorm_0@kernel_rms_swiglu__rms_norm_fold_nogain_2@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_syn_kern_fuse_silu__kern_fuse_rmsnorm_1@kernel_rms_swiglu__rms_norm_fold_nogain_5@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_swiglu__kern_square_strength_7@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_silu__kern_fuse_rmsnorm_0@kernel_rms_swiglu__kern_square_strength_1@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_silu__kern_fuse_rmsnorm_1@kernel_rms_swiglu__kern_square_strength_4@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_silu__kern_fuse_rmsnorm_0@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_silu__kern_fuse_rmsnorm_1@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_silu__rms_norm_fold_nogain_4@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_silu__rms_norm_fold_nogain_5@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_mul3__kern_fuse_rmsnorm_9@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mul3__kern_fuse_rmsnorm_10@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mul3__rms_norm_fold_nogain_13@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_mul3__rms_norm_fold_nogain_14@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_mul3__kern_fuse_rmsnorm_15@kernel_rms_swiglu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mul3__rms_norm_fold_nogain_18@kernel_rms_swiglu` | synth-composite | unverified | no | oracle: corpus:kernel_rms_swiglu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_silu__kern_square_strength_2@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_silu__kern_square_strength_3@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_mul3__kern_square_strength_11@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_mul3__kern_square_strength_12@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_mul3__kern_square_strength_16@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_mul3__kern_square_strength_17@kernel_rms_swiglu` | synth-composite | useful | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_layernorm__kern_fuse_mm_gelu_0@kernel_ln_mm_gelu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_rmsnorm__kern_fuse_mm_gelu_1@kernel_ln_mm_gelu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mm_gelu__rms_norm_fold_nogain_5@kernel_ln_mm_gelu` | synth-composite | unverified | no | oracle: corpus:kernel_ln_mm_gelu: skip (KeyError: "KernelSink cannot lower op 'rms_norm' — not in supported_ops") — inconclusive | — |
| `syn_kern_fuse_mm_gelu__kern_square_strength_4@kernel_ln_mm_gelu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_square_strength_6@kernel_ln_mm_gelu` | synth-composite | useful | yes | cleared | known |
| `syn_naturality_scalar_rev__kern_fuse_mm_bias_0@kernel_mm_sb_gelu` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__kern_fuse_mm_scale_2@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_0@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__naturality_scalar_rev_3@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_4@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `recognize:softmax` | pattern-recognition | known | yes | cleared | unpaid |
| `syn_kern_fuse_softmax__naturality_scalar_rev_1@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__naturality_scalar_rev_5@kernel_attn_residual` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_bias_2@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_bias_6@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_1@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_5@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_softmax__naturality_scalar_rev_3@kernel_softmax_masked__kern_fuse_mm_bias_0@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_syn_softmax_fold__naturality_scalar_rev_7@kernel_softmax_masked__kern_fuse_mm_bias_1@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_addmul_0@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_addmul_4@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__naturality_scalar_rev_3@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__naturality_scalar_rev_7@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `syn_naturality_scalar_rev__kern_fuse_mm_bias_8@kernel_softmax_masked` | synth-composite | useful | yes | cleared | known |
| `grammar:exp_distribute` | algebraic-grammar | true | no | pay: no verified cost-improving extraction | — |
| `grammar:FALSE_exp_add` | algebraic-grammar | false | no | oracle: corpus:kernel_softmax_masked: unequal (max_abs 5.66e+11) | — |
| `syn_naturality_scalar_rev__kern_fuse_mm_bias_0@kernel_mm_scale_bias` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__kern_fuse_mm_scale_2@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_0@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__naturality_scalar_rev_3@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_4@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__naturality_scalar_rev_1@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__naturality_scalar_rev_5@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__kern_fuse_mm_scale_2@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_0@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__naturality_scalar_rev_3@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_4@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__naturality_scalar_rev_1@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__naturality_scalar_rev_5@kernel_sdpa_cap` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_square_strength__kern_fuse_addmul_0@kernel_strength__kern_div_recip_0@kernel_strength` | synth-composite | useful | yes | cleared | known |
| `syn_kern_square_strength__kern_fuse_addmul_0@kernel_strength` | synth-composite | useful | yes | cleared | known |
| `syn_kern_square_strength__kern_fuse_mul3_0@kernel_powscale` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_square_strength__kern_fuse_mul3_0@kernel_strength_mul__kern_div_recip_0@kernel_strength_mul` | synth-composite | useful | yes | cleared | known |
| `syn_kern_square_strength__kern_fuse_mul3_0@kernel_strength_mul` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_silu__kern_fuse_silu_0@kernel_mixed` | synth-composite | useful | yes | cleared | known |
| `mixed:mul_unsqueeze_l_id` | census-mixed-view | known | no | oracle: corpus:kernel_decayed_pair: skip (RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 1) — inconclusive | — |
| `mixed:mul_unsqueeze_l_w` | census-mixed-view | known | no | oracle: corpus:kernel_decayed_pair: skip (RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 1) — inconclusive | — |
| `syn_kern_fuse_mm_bias__kern_fuse_om_decayed_0@kernel_decayed_chunks` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan__kern_fuse_mm_scale_0@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan_1@kernel_chunk_scan__kern_fuse_mm_scale_0@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_8@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_2@kernel_chunk_scan__kern_fuse_om_decayed_1@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan_1@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_2@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_6@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan__kern_div_recip_9@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_div_recip_4@kernel_chunk_scan__kern_fuse_mm_scale_10@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_fuse_om_decayed_4@kernel_chunk_scan__kern_div_recip_3@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_div_recip_5@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan__kern_div_recip_7@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan__kern_div_recip_4@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_fuse_om_decayed_4@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_1@kernel_decayed_div` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_div_recip_0@kernel_decayed_div` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan3__kern_fuse_mm_scale_0@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_mm_bias__kern_div_recip_1@kernel_chunk_scan3__kern_fuse_mm_scale_1@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mm_bias__kern_div_recip_1@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_10@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_11@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_12@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan3_2@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__syn_kern_fuse_mm_bias__kern_div_recip_1@kernel_chunk_scan3_3@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__syn_kern_div_recip__kern_fuse_mm_scale_10@kernel_chunk_scan3_4@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__syn_kern_div_recip__kern_fuse_mm_scale_11@kernel_chunk_scan3_5@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__syn_kern_div_recip__kern_fuse_mm_scale_12@kernel_chunk_scan3_6@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan_3__kern_div_recip_7@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan_3__kern_div_recip_8@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan_3__kern_div_recip_9@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__kern_fuse_mm_bias_7@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3__kern_fuse_mm_bias_8@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan_3__kern_fuse_mm_bias_5@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan_3__kern_fuse_mm_bias_6@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_2@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_3@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_3_4@kernel_chunk_scan3` | synth-composite | useful | yes | cleared | known |
