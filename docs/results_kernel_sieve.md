# Kernel novelty sieve

> Snapshot note: this run (2026-10-22) predates `kern_fuse_sdpa_mask` /
> `kern_fuse_sdpa_gqa` — those two laws are absent from the table below.

The codegen port of the novelty sieve — the automatic three-gate filter between a shipped ``KERNEL_RULES`` law and the human-review queue.  **cert** = the law's own rewrite carries a standalone-replayable certificate on its instance (`verify_law` + `verify_certificate`); **derivable** = the known bank — `KERNEL_RULES` U the kernel prior-art bank (`KERNEL_PRIOR_ART_RULES`, the codegen vocabulary) U the shared substrate surface (`default_rules` + `ALL_RULES_WITH_LAYOUT` + `ATTENTION` + scan monoids + the prior-art FUSION/CATEGORICAL slice) — minus the candidate and its alpha-siblings, already proves the instance; **beat** = on some registered `bench.discovery.kernel` case the certified extraction under the full kernel bank lands strictly below the best extraction the rest of the bank reaches on the same term (the floor — itself never worse than spelled) AND the derivation certificate invokes the candidate.  A law is a novelty iff cert ∧ beat ∧ ¬derivable.  `truth` is the fp64 `KernelSink.verify` oracle on the instance — an `unequal` refutes outright.

- date: 2026-10-22 23:09 UTC · git: `e5f7e22` (dirty) · torch 2.14.1+cu130 on cuda · bank 313 rules
- verdicts: known=26

| law | family | prov | instance | truth | cert | derivable | beat | verdict |
|---|---|---|---|---|---|---|---|---|
| `kern_fuse_silu` | fusion | human | corpus:kernel_silu | equal | yes | yes (pa_kf_silu_expand) | — | **known** |
| `kern_fuse_softsign` | fusion | human | corpus:kernel_softsign | equal | yes | yes (pa_kf_softsign_expand) | — | **known** |
| `kern_fuse_addmul` | fusion | human | corpus:kernel_fma | equal | yes | yes (pa_kf_addmul_expand) | — | **known** |
| `kern_fuse_swiglu` | fusion | human | corpus:kernel_swiglu | equal | yes | yes (pa_kf_swiglu_expand) | — | **known** |
| `kern_fuse_silu_sigmoid` | fusion | human | corpus:kernel_mixed | equal | yes | yes (pa_kf_silu_sigmoid_expand) | — | **known** |
| `kern_fuse_addmul_tanh` | fusion | human | corpus:kernel_fma | equal | yes | yes (pa_kf_addmul_tanh_expand) | — | **known** |
| `kern_fuse_addmul2` | fusion | human | corpus:kernel_addmul2 | equal | yes | yes (pa_kf_addmul2_expand) | — | **known** |
| `kern_fuse_mul3` | fusion | human | corpus:kernel_swiglu | equal | yes | yes (pa_kf_mul3_expand) | — | **known** |
| `kern_fuse_softmax` | fusion | human | corpus:kernel_softmax | equal | yes | yes (pa_kf_softmax_expand) | — | **known** |
| `kern_fuse_softmax_mm` | fusion | human | corpus:kernel_softmax_mm | equal | yes | yes (pa_kf_softmax_mm_expand) | — | **known** |
| `kern_fuse_sdpa` | fusion | human | corpus:kernel_sdpa | equal | yes | yes (pa_kf_sdpa_expand) | — | **known** |
| `kern_fuse_sdpa_cap` | fusion | human | corpus:kernel_sdpa_cap | equal | yes | yes (pa_kf_sdpa_cap_expand) | — | **known** |
| `kern_fuse_layernorm` | fusion | human | corpus:kernel_layernorm | equal | yes | yes (pa_kf_layernorm_expand) | — | **known** |
| `kern_fuse_rmsnorm` | fusion | human | corpus:kernel_layernorm | equal | yes | yes (pa_kf_rmsnorm_expand) | — | **known** |
| `kern_fuse_mm_bias` | fusion | human | corpus:kernel_mm_bias | equal | yes | yes (pa_kf_mm_bias_expand) | — | **known** |
| `kern_fuse_mm_scale` | fusion | human | corpus:kernel_mm_scale_bias | equal | yes | yes (pa_kf_mm_scale_expand) | — | **known** |
| `kern_fuse_mm_scale_bias` | fusion | human | corpus:kernel_mm_scale_bias | equal | yes | yes (pa_kf_mm_sb_expand) | — | **known** |
| `kern_fuse_mm_gelu` | fusion | human | corpus:kernel_mm_gelu | equal | yes | yes (pa_kf_mm_gelu_expand) | — | **known** |
| `kern_fuse_om_decayed` | fusion | human | corpus:kernel_decayed | equal | yes | yes (pa_om_decayed_expand) | — | **known** |
| `kern_fuse_chunk_scan` | schedule | human | corpus:kernel_chunk_scan | equal | yes | yes (pa_om_chunk_scan_expand_2) | — | **known** |
| `kern_fuse_chunk_scan_3` | schedule | human | corpus:xcorpus:scan_3 | equal | yes | yes (pa_om_chunk_scan_expand_3) | — | **known** |
| `kern_fuse_chunk_scan_4` | schedule | human | corpus:xcorpus:scan_4 | equal | yes | yes (pa_om_chunk_scan_expand_4) | — | **known** |
| `kern_fuse_chunk_scan_5` | schedule | human | corpus:xcorpus:scan_5 | equal | yes | yes (pa_om_chunk_scan_expand_5) | — | **known** |
| `kern_fuse_chunk_scan_6` | schedule | human | corpus:xcorpus:scan_6 | equal | yes | yes (pa_om_chunk_scan_expand_6) | — | **known** |
| `kern_square_strength` | strength | human | corpus:kernel_layernorm | equal | yes | yes (pow_to_square, square_expand) | — | **known** |
| `kern_div_recip` | strength | human | corpus:kernel_strength | equal | yes | yes (pa_const_fold_recip, pa_div_const_recip) | — | **known** |

## Review queue

**Empty.** Every candidate either fails a gate or is a theorem of the bank — nothing currently needs a reviewer's eyes.

## Per-law detail

### `kern_fuse_silu` — known

- instance (corpus:kernel_silu): `(mul x, (sigmoid x))` → `(kf_silu x, x)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_silu_expand — fixed_point
- siblings excluded: kern_fuse_silu
- beat: — — —

### `kern_fuse_softsign` — known

- instance (corpus:kernel_softsign): `(div x, (add (abs x), 1))` → `(kf_softsign x, x, 1)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_softsign_expand — fixed_point
- siblings excluded: kern_fuse_softsign
- beat: — — —

### `kern_fuse_addmul` — known

- instance (corpus:kernel_fma): `(add (mul x, y), (tanh y))` → `(kf_addmul x, y, (tanh y))`
- truth: equal · cert: yes
- derivable: yes via pa_kf_addmul_expand — fixed_point
- siblings excluded: kern_fuse_addmul
- beat: — — —

### `kern_fuse_swiglu` — known

- instance (corpus:kernel_swiglu): `(mul (mul x, (sigmoid x)), y)` → `(kf_swiglu x, y)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_swiglu_expand — fixed_point
- siblings excluded: kern_fuse_swiglu
- beat: — — —

### `kern_fuse_silu_sigmoid` — known

- instance (corpus:kernel_mixed): `(mul (mul x, (sigmoid x)), (sigmoid y))` → `(kf_silu_sigmoid x, y)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_silu_sigmoid_expand — fixed_point
- siblings excluded: kern_fuse_silu_sigmoid
- beat: — — —

### `kern_fuse_addmul_tanh` — known

- instance (corpus:kernel_fma): `(add (mul x, y), (tanh y))` → `(kf_addmul_tanh x, y, y)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_addmul_tanh_expand — fixed_point
- siblings excluded: kern_fuse_addmul_tanh
- beat: — — —

### `kern_fuse_addmul2` — known

- instance (corpus:kernel_addmul2): `(add (mul x, y), (mul y, b))` → `(kf_addmul2 x, y, y, b)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_addmul2_expand — fixed_point
- siblings excluded: kern_fuse_addmul2
- beat: — — —

### `kern_fuse_mul3` — known

- instance (corpus:kernel_swiglu): `(mul (mul x, (sigmoid x)), y)` → `(kf_mul3 x, (sigmoid x), y)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_mul3_expand — fixed_point
- siblings excluded: kern_fuse_mul3
- beat: — — —

### `kern_fuse_softmax` — known

- instance (corpus:kernel_softmax): `(div (exp x), (sum (exp x), dim=-1, keepdim=True))` → `(kf_softmax x, dim=-1)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_softmax_expand — fixed_point
- siblings excluded: kern_fuse_softmax
- beat: — — —

### `kern_fuse_softmax_mm` — known

- instance (corpus:kernel_softmax_mm): `(matmul (div (exp (tanh (matmul x, w))), (sum (exp (tanh (matmul x, w))), dim=-1, keepdim=True)), w)` → `(kf_softmax_mm (tanh (matmul x, w)), w, dim=-1)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_softmax_mm_expand — fixed_point
- siblings excluded: kern_fuse_softmax_mm
- beat: — — —

### `kern_fuse_sdpa` — known

- instance (corpus:kernel_sdpa): `(matmul (div (exp (mul (matmul x, w), 0.25)), (sum (exp (mul (matmul x, w), 0.25)), dim=-1, keepdim=True)), w)` → `(kf_sdpa x, w, 0.25, w, dim=-1, cap=False)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_sdpa_expand — fixed_point
- siblings excluded: kern_fuse_sdpa
- beat: — — —

### `kern_fuse_sdpa_cap` — known

- instance (corpus:kernel_sdpa_cap): `(matmul (div (exp (tanh (mul (matmul x, w), 0.25))), (sum (exp (tanh (mul (matmul x, w), 0.25))), dim=-1, keepdim=True)), w)` → `(kf_sdpa x, w, 0.25, w, dim=-1, cap=True)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_sdpa_cap_expand — fixed_point
- siblings excluded: kern_fuse_sdpa_cap
- beat: — — —

### `kern_fuse_layernorm` — known

- instance (corpus:kernel_layernorm): `(mul (sub x, (mean x, dim=-1, keepdim=True)), (rsqrt (add (mean (pow (sub x, (mean x, dim=-1, keepdim=True)), 2), dim=-1, keepdim=True), 1e-05)))` → `(kf_layernorm x, dim=-1, eps=1e-05)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_layernorm_expand — fixed_point
- siblings excluded: kern_fuse_layernorm
- beat: — — —

### `kern_fuse_rmsnorm` — known

- instance (corpus:kernel_layernorm): `(mul (sub x, (mean x, dim=-1, keepdim=True)), (rsqrt (add (mean (pow (sub x, (mean x, dim=-1, keepdim=True)), 2), dim=-1, keepdim=True), 1e-05)))` → `(kf_rmsnorm (sub x, (mean x, dim=-1, keepdim=True)), dim=-1, eps=1e-05)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_rmsnorm_expand — fixed_point
- siblings excluded: kern_fuse_rmsnorm
- beat: — — —

### `kern_fuse_mm_bias` — known

- instance (corpus:kernel_mm_bias): `(add (matmul x, w), b)` → `(kf_mm_bias x, w, b)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_mm_bias_expand — fixed_point
- siblings excluded: kern_fuse_mm_bias
- beat: — — —

### `kern_fuse_mm_scale` — known

- instance (corpus:kernel_mm_scale_bias): `(mul (matmul x, w), 0.5)` → `(kf_mm_scale x, w, 0.5)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_mm_scale_expand — fixed_point
- siblings excluded: kern_fuse_mm_scale
- beat: — — —

### `kern_fuse_mm_scale_bias` — known

- instance (corpus:kernel_mm_scale_bias): `(add (mul (matmul x, w), 0.5), b)` → `(kf_mm_sb x, w, 0.5, b)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_mm_sb_expand — fixed_point
- siblings excluded: kern_fuse_mm_scale_bias
- beat: — — —

### `kern_fuse_mm_gelu` — known

- instance (corpus:kernel_mm_gelu): `(gelu (matmul x, w))` → `(kf_mm_gelu x, w)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_mm_gelu_expand — fixed_point
- siblings excluded: kern_fuse_mm_gelu
- beat: — — —

### `kern_fuse_om_decayed` — known

- instance (corpus:kernel_decayed): `(matmul (mul s, d), v)` → `(om_decayed s, d, v)`
- truth: equal · cert: yes
- derivable: yes via pa_om_decayed_expand — fixed_point
- siblings excluded: kern_fuse_om_decayed
- beat: — — —

### `kern_fuse_chunk_scan` — known

- instance (corpus:kernel_chunk_scan): `(concat (matmul (mul (split sc, sizes=(64, 64), dim=1, index=0), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=0)), (add (matmul (mul (split sc, sizes=(64, 64), dim=1, index=1), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=1)), (mul (unsqueeze (decay_vec (sigmoid gc), t=64), dim=-1), (div (matmul (split sc, sizes=(64, 64), dim=1, index=1), (matmul (split sc, sizes=(64, 64), dim=1, index=0), (split vc, sizes=(64, 64), dim=1, index=0))), 2))), dim=-2)` → `(om_chunk_scan (matmul (mul (split sc, sizes=(64, 64), dim=1, index=0), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=0)), (add (matmul (mul (split sc, sizes=(64, 64), dim=1, index=1), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=1)), (mul (unsqueeze (decay_vec (sigmoid gc), t=64), dim=-1), (div (matmul (split sc, sizes=(64, 64), dim=1, index=1), (matmul (split sc, sizes=(64, 64), dim=1, index=0), (split vc, sizes=(64, 64), dim=1, index=0))), 2))), dim=-2)`
- truth: equal · cert: yes
- derivable: yes via pa_om_chunk_scan_expand_2 — fixed_point
- siblings excluded: kern_fuse_chunk_scan
- beat: — — —

### `kern_fuse_chunk_scan_3` — known

- instance (corpus:xcorpus:scan_3): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: yes via pa_om_chunk_scan_expand_3 — fixed_point
- siblings excluded: kern_fuse_chunk_scan_3
- beat: — — —

### `kern_fuse_chunk_scan_4` — known

- instance (corpus:xcorpus:scan_4): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: yes via pa_om_chunk_scan_expand_4 — fixed_point
- siblings excluded: kern_fuse_chunk_scan_4
- beat: — — —

### `kern_fuse_chunk_scan_5` — known

- instance (corpus:xcorpus:scan_5): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: yes via pa_om_chunk_scan_expand_5 — max_nodes
- siblings excluded: kern_fuse_chunk_scan_5
- beat: — — —

### `kern_fuse_chunk_scan_6` — known

- instance (corpus:xcorpus:scan_6): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), (add (add (add (add (add (div (om_decayed s5, d5, v5), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a3, v3)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a4, v4)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), (add (add (add (add (add (div (om_decayed s5, d5, v5), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a3, v3)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a4, v4)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: yes via pa_om_chunk_scan_expand_6 — max_nodes
- siblings excluded: kern_fuse_chunk_scan_6
- beat: — — —

### `kern_square_strength` — known

- instance (corpus:kernel_layernorm): `(pow (sub x, (mean x, dim=-1, keepdim=True)), 2)` → `(mul (sub x, (mean x, dim=-1, keepdim=True)), (sub x, (mean x, dim=-1, keepdim=True)))`
- truth: equal · cert: yes
- derivable: yes via pow_to_square, square_expand — fixed_point
- siblings excluded: kern_square_strength
- beat: — — —

### `kern_div_recip` — known

- instance (corpus:kernel_strength): `(div y, 4.0)` → `(mul y, 0.25)`
- truth: equal · cert: yes
- derivable: yes via pa_const_fold_recip, pa_div_const_recip — fixed_point
- siblings excluded: kern_div_recip
- beat: — — —

