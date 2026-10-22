# Kernel overnight discovery run

Episodes -> mint -> admit -> classify on the kernel corpus.  Each episode's arena pool is minted *from the case's own term*: the shared corpus-census proposals that bind it plus the ``syn_*`` pair-composites ``meta.synthesize_rules`` mints over ``KERNEL_RULES`` (+ the substrate's guarded parents) seeded on the term.  Every mint faces the emit gate (strict LHS bind + instantiated RHS must denote).  Evaluation binds a firing site, verifies the concrete pair through ``KernelSink`` (eager bodies, fp64) and prices the firing in ``kernel_cost``.  The admission gauntlet is the kernel referee — ``oracle`` (instance evaluates ``equal``), ``pay`` (the rule alone must extract a *verified* term strictly below the spelled cost on a registered case), ``sweep`` (all bound corpus sites equal), ``cert`` (the instance carries a standalone-replayable certificate).  Admitted rules are classified by the novelty sieve's gates — **review** (cert ∧ beat ∧ ¬derivable: the bank cannot prove it) is the machine-original class.

- date: 2026-10-22 22:49 UTC · git: `0cdc3f0` (dirty) · torch 2.14.1+cu130 on cuda
- protocol: `{"budget": 24, "cases": ["kernel_silu", "kernel_swiglu", "kernel_softmax", "kernel_layernorm", "kernel_sdpa", "kernel_strength", "kernel_powscale", "kernel_addmul2", "kernel_mixed", "kernel_decayed_pair", "kernel_chunk_scan"], "dry": false, "episodes": 10, "fuel": 256, "mint": "kernel-corpus-compose", "page": 24, "player": "playbook", "referee": "full", "rounds": 2, "seed": 0, "vocab": "hand"}`
- episodes: 11 · evaluated: 34 · minted: 31 (27 composites, 24 guarded) · admitted: 28 · wall: 16.7s
- arena verdicts: known=7, useful=27

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 2 |
| pay | 1 |
| **admitted** | 28 |

## Novelty

| verdict | count |
|---|---|
| known | 27 |
| unpaid | 1 |

## Machine-original finds (sieve `review`)

**None.** Every admitted mint was already a theorem of the known bank — or nothing survived the gauntlet.  The funnel above is the measurement: composed kernel rules are derivable by construction when both parents are in the bank, so the honest expectation is refusals at ``pay`` (a composite never beats the floor its own chain reaches) and ``known`` verdicts at classify.

## Candidates

| candidate | family | arena | admitted | reason | novelty |
|---|---|---|---|---|---|
| `recognize:softmax` | pattern-recognition | known | yes | cleared | unpaid |
| `grammar:sub_to_add_dup` | algebraic-grammar | known | no | pay: no verified cost-improving extraction | — |
| `syn_kern_fuse_softmax_mm__kern_fuse_mm_scale_2@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__kern_fuse_mm_scale_0@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax_mm__naturality_scalar_rev_3@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__kern_fuse_mm_scale_4@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_softmax__naturality_scalar_rev_1@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_softmax_fold__naturality_scalar_rev_5@kernel_sdpa` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_square_strength__kern_fuse_addmul_0@kernel_strength__kern_div_recip_0@kernel_strength` | synth-composite | useful | yes | cleared | known |
| `syn_kern_square_strength__kern_fuse_addmul_0@kernel_strength` | synth-composite | useful | yes | cleared | known |
| `syn_kern_square_strength__kern_fuse_mul3_0@kernel_powscale` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_silu__kern_fuse_silu_0@kernel_mixed` | synth-composite | useful | yes | cleared | known |
| `mixed:mul_unsqueeze_l_id` | census-mixed-view | known | no | oracle: corpus:kernel_decayed_pair: skip (RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 1) — inconclusive | — |
| `mixed:mul_unsqueeze_l_w` | census-mixed-view | known | no | oracle: corpus:kernel_decayed_pair: skip (RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 1) — inconclusive | — |
| `syn_syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan__kern_fuse_mm_scale_0@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_8@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_fuse_mm_bias__kern_div_recip_0@kernel_chunk_scan_1@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_2@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan__syn_kern_div_recip__kern_fuse_mm_scale_5@kernel_chunk_scan_6@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan__kern_div_recip_9@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_chunk_scan__kern_div_recip_4@kernel_chunk_scan__kern_fuse_mm_scale_10@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_div_recip_5@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan__kern_div_recip_7@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan__kern_div_recip_4@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_chunk_scan__kern_fuse_mm_bias_3@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan__kern_fuse_om_decayed_4@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_1@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
| `syn_kern_fuse_om_decayed__kern_fuse_chunk_scan_2@kernel_chunk_scan` | synth-composite | useful | yes | cleared | known |
