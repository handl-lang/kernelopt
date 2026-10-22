# Kernel novelty sieve

The codegen port of the novelty sieve — the automatic three-gate filter between a shipped ``KERNEL_RULES`` law and the human-review queue.  **cert** = the law's own rewrite carries a standalone-replayable certificate on its instance (`verify_law` + `verify_certificate`); **derivable** = the known bank — `KERNEL_RULES` ∪ the shared substrate surface (`default_rules` + `ALL_RULES_WITH_LAYOUT` + `ATTENTION` + scan monoids + the prior-art FUSION/CATEGORICAL slice) — minus the candidate and its alpha-siblings, already proves the instance; **beat** = on some registered `bench.discovery.kernel` case the certified extraction under the full kernel bank lands strictly below the best extraction the rest of the bank reaches on the same term (the floor — itself never worse than spelled) AND the derivation certificate invokes the candidate.  A law is a novelty iff cert ∧ beat ∧ ¬derivable.  `truth` is the fp64 `KernelSink.verify` oracle on the instance — an `unequal` refutes outright.

- date: 2026-10-21 23:46 UTC · git: `72022cb` (dirty) · torch 2.14.1+cu130 on cuda · bank 264 rules
- verdicts: known=1, review=20, unmeasured=4, unpaid=1

| law | family | prov | instance | truth | cert | derivable | beat | verdict |
|---|---|---|---|---|---|---|---|---|
| `kern_fuse_silu` | fusion | human | corpus:kernel_silu | equal | yes | no | yes | **review** |
| `kern_fuse_softsign` | fusion | human | corpus:kernel_softsign | equal | yes | no | yes | **review** |
| `kern_fuse_addmul` | fusion | human | corpus:kernel_fma | equal | yes | no | no | **unpaid** |
| `kern_fuse_swiglu` | fusion | human | corpus:kernel_swiglu | equal | yes | no | yes | **review** |
| `kern_fuse_silu_sigmoid` | fusion | human | corpus:kernel_mixed | equal | yes | no | yes | **review** |
| `kern_fuse_addmul_tanh` | fusion | human | corpus:kernel_fma | equal | yes | no | yes | **review** |
| `kern_fuse_addmul2` | fusion | human | corpus:kernel_addmul2 | equal | yes | no | yes | **review** |
| `kern_fuse_mul3` | fusion | human | corpus:kernel_swiglu | equal | yes | no | yes | **review** |
| `kern_fuse_softmax` | fusion | human | corpus:kernel_softmax | equal | yes | no | yes | **review** |
| `kern_fuse_softmax_mm` | fusion | human | corpus:kernel_softmax_mm | equal | yes | no | yes | **review** |
| `kern_fuse_sdpa` | fusion | human | corpus:kernel_sdpa | equal | yes | no | yes | **review** |
| `kern_fuse_sdpa_cap` | fusion | human | corpus:kernel_sdpa_cap | equal | yes | no | yes | **review** |
| `kern_fuse_layernorm` | fusion | human | corpus:kernel_layernorm | equal | yes | no | yes | **review** |
| `kern_fuse_rmsnorm` | fusion | human | corpus:kernel_layernorm | equal | yes | no | yes | **review** |
| `kern_fuse_mm_bias` | fusion | human | corpus:kernel_mm_bias | equal | yes | no | yes | **review** |
| `kern_fuse_mm_scale` | fusion | human | corpus:kernel_mm_scale_bias | equal | yes | no | yes | **review** |
| `kern_fuse_mm_scale_bias` | fusion | human | corpus:kernel_mm_scale_bias | equal | yes | no | yes | **review** |
| `kern_fuse_mm_gelu` | fusion | human | corpus:kernel_mm_gelu | equal | yes | no | yes | **review** |
| `kern_fuse_om_decayed` | fusion | human | corpus:kernel_decayed | equal | yes | no | yes | **review** |
| `kern_fuse_chunk_scan` | schedule | human | corpus:kernel_chunk_scan | equal | yes | no | yes | **review** |
| `kern_fuse_chunk_scan_3` | schedule | human | corpus:xcorpus:scan_3 | equal | yes | no | — | **unmeasured** |
| `kern_fuse_chunk_scan_4` | schedule | human | corpus:xcorpus:scan_4 | equal | yes | no | — | **unmeasured** |
| `kern_fuse_chunk_scan_5` | schedule | human | corpus:xcorpus:scan_5 | equal | yes | no | — | **unmeasured** |
| `kern_fuse_chunk_scan_6` | schedule | human | corpus:xcorpus:scan_6 | equal | yes | no | — | **unmeasured** |
| `kern_square_strength` | strength | human | corpus:kernel_layernorm | equal | yes | yes (pow_to_square, square_expand) | — | **known** |
| `kern_div_recip` | strength | human | corpus:kernel_strength | equal | yes | no | yes | **review** |

## Review queue

Certified, measured-paying, and underivable from the known bank — these are for a human to look at:

- `kern_fuse_silu` — kernel_silu (3 → 1 vs floor 3)
- `kern_fuse_softsign` — kernel_softsign (4 → 1 vs floor 4)
- `kern_fuse_swiglu` — kernel_swiglu (4 → 1 vs floor 2)
- `kern_fuse_silu_sigmoid` — kernel_mixed (6 → 1 vs floor 2)
- `kern_fuse_addmul_tanh` — kernel_fma (4 → 1 vs floor 3)
- `kern_fuse_addmul2` — kernel_strength (7 → 1 vs floor 2), kernel_addmul2 (3 → 1 vs floor 2)
- `kern_fuse_mul3` — kernel_powscale (5 → 1 vs floor 2)
- `kern_fuse_softmax` — kernel_softmax (7 → 1 vs floor 7)
- `kern_fuse_softmax_mm` — kernel_softmax_mm (14 → 4 vs floor 5)
- `kern_fuse_sdpa` — kernel_sdpa (12 → 1 vs floor 2)
- `kern_fuse_sdpa_cap` — kernel_sdpa_cap (16 → 1 vs floor 4)
- `kern_fuse_layernorm` — kernel_layernorm (13 → 1 vs floor 3)
- `kern_fuse_rmsnorm` — kernel_rmsnorm (9 → 1 vs floor 6)
- `kern_fuse_mm_bias` — kernel_mm_bias (2 → 1 vs floor 2)
- `kern_fuse_mm_scale` — kernel_mm_scale (2 → 1 vs floor 2)
- `kern_fuse_mm_scale_bias` — kernel_mm_scale_bias (3 → 1 vs floor 2)
- `kern_fuse_mm_gelu` — kernel_mm_gelu (3 → 1 vs floor 3)
- `kern_fuse_om_decayed` — kernel_decayed (2 → 1 vs floor 2), kernel_decayed_mask (5 → 4 vs floor 5), kernel_decayed_intra (6 → 5 vs floor 6), kernel_decayed_chunks (11 → 9 vs floor 10), kernel_decayed_t512 (2 → 1 vs floor 2), kernel_decayed_t1024 (2 → 1 vs floor 2), kernel_decayed_t2048 (2 → 1 vs floor 2)
- `kern_fuse_chunk_scan` — kernel_chunk_scan (20 → 15 vs floor 16)
- `kern_div_recip` — kernel_strength (7 → 1 vs floor 3), kernel_chunk_scan (20 → 15 vs floor 17)

## Per-law detail

### `kern_fuse_silu` — review

- instance (corpus:kernel_silu): `(mul x, (sigmoid x))` → `(kf_silu x, x)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_silu
- beat: yes — 3 case(s) run
  - `kernel_silu`: cert yes, verify yes, fired 1, in-deriv yes, cost 3 → 1 vs floor 3 → beat yes
  - `kernel_swiglu`: cert yes, verify yes, fired 1, in-deriv no, cost 4 → 1 vs floor 1 → beat no
  - `kernel_mixed`: cert yes, verify yes, fired 2, in-deriv no, cost 6 → 1 vs floor 1 → beat no

### `kern_fuse_softsign` — review

- instance (corpus:kernel_softsign): `(div x, (add (abs x), 1))` → `(kf_softsign x, x, 1)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_softsign
- beat: yes — 1 case(s) run
  - `kernel_softsign`: cert yes, verify yes, fired 1, in-deriv yes, cost 4 → 1 vs floor 4 → beat yes

### `kern_fuse_addmul` — unpaid

- instance (corpus:kernel_fma): `(add (mul x, y), (tanh y))` → `(kf_addmul x, y, (tanh y))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_addmul
- beat: no — 4 case(s) run
  - `kernel_fma`: cert yes, verify yes, fired 1, in-deriv no, cost 4 → 1 vs floor 1 → beat no
  - `kernel_mm_scale_bias`: cert yes, verify yes, fired 1, in-deriv no, cost 3 → 1 vs floor 1 → beat no
  - `kernel_strength`: cert yes, verify yes, fired 1, in-deriv no, cost 7 → 1 vs floor 1 → beat no
  - `kernel_addmul2`: cert yes, verify yes, fired 1, in-deriv no, cost 3 → 1 vs floor 1 → beat no

### `kern_fuse_swiglu` — review

- instance (corpus:kernel_swiglu): `(mul (mul x, (sigmoid x)), y)` → `(kf_swiglu x, y)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_swiglu
- beat: yes — 2 case(s) run
  - `kernel_swiglu`: cert yes, verify yes, fired 1, in-deriv yes, cost 4 → 1 vs floor 2 → beat yes
  - `kernel_mixed`: cert yes, verify yes, fired 1, in-deriv no, cost 6 → 1 vs floor 1 → beat no

### `kern_fuse_silu_sigmoid` — review

- instance (corpus:kernel_mixed): `(mul (mul x, (sigmoid x)), (sigmoid y))` → `(kf_silu_sigmoid x, y)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_silu_sigmoid
- beat: yes — 1 case(s) run
  - `kernel_mixed`: cert yes, verify yes, fired 1, in-deriv yes, cost 6 → 1 vs floor 2 → beat yes

### `kern_fuse_addmul_tanh` — review

- instance (corpus:kernel_fma): `(add (mul x, y), (tanh y))` → `(kf_addmul_tanh x, y, y)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_addmul_tanh
- beat: yes — 1 case(s) run
  - `kernel_fma`: cert yes, verify yes, fired 1, in-deriv yes, cost 4 → 1 vs floor 3 → beat yes

### `kern_fuse_addmul2` — review

- instance (corpus:kernel_addmul2): `(add (mul x, y), (mul y, b))` → `(kf_addmul2 x, y, y, b)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_addmul2
- beat: yes — 2 case(s) run
  - `kernel_strength`: cert yes, verify yes, fired 1, in-deriv yes, cost 7 → 1 vs floor 2 → beat yes
  - `kernel_addmul2`: cert yes, verify yes, fired 1, in-deriv yes, cost 3 → 1 vs floor 2 → beat yes

### `kern_fuse_mul3` — review

- instance (corpus:kernel_swiglu): `(mul (mul x, (sigmoid x)), y)` → `(kf_mul3 x, (sigmoid x), y)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_mul3
- beat: yes — 3 case(s) run
  - `kernel_powscale`: cert yes, verify yes, fired 1, in-deriv yes, cost 5 → 1 vs floor 2 → beat yes
  - `kernel_swiglu`: cert yes, verify yes, fired 1, in-deriv no, cost 4 → 1 vs floor 1 → beat no
  - `kernel_mixed`: cert yes, verify yes, fired 1, in-deriv no, cost 6 → 1 vs floor 1 → beat no

### `kern_fuse_softmax` — review

- instance (corpus:kernel_softmax): `(div (exp x), (sum (exp x), dim=-1, keepdim=True))` → `(kf_softmax x, dim=-1)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_softmax
- beat: yes — 4 case(s) run
  - `kernel_softmax`: cert yes, verify yes, fired 1, in-deriv yes, cost 7 → 1 vs floor 7 → beat yes
  - `kernel_softmax_mm`: cert yes, verify yes, fired 1, in-deriv no, cost 14 → 4 vs floor 4 → beat no
  - `kernel_sdpa`: cert yes, verify yes, fired 1, in-deriv no, cost 12 → 1 vs floor 1 → beat no
  - `kernel_sdpa_cap`: cert yes, verify yes, fired 1, in-deriv no, cost 16 → 1 vs floor 1 → beat no

### `kern_fuse_softmax_mm` — review

- instance (corpus:kernel_softmax_mm): `(matmul (div (exp (tanh (matmul x, w))), (sum (exp (tanh (matmul x, w))), dim=-1, keepdim=True)), w)` → `(kf_softmax_mm (tanh (matmul x, w)), w, dim=-1)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_softmax_mm
- beat: yes — 3 case(s) run
  - `kernel_softmax_mm`: cert yes, verify yes, fired 1, in-deriv yes, cost 14 → 4 vs floor 5 → beat yes
  - `kernel_sdpa`: cert yes, verify yes, fired 1, in-deriv no, cost 12 → 1 vs floor 1 → beat no
  - `kernel_sdpa_cap`: cert yes, verify yes, fired 1, in-deriv no, cost 16 → 1 vs floor 1 → beat no

### `kern_fuse_sdpa` — review

- instance (corpus:kernel_sdpa): `(matmul (div (exp (mul (matmul x, w), 0.25)), (sum (exp (mul (matmul x, w), 0.25)), dim=-1, keepdim=True)), w)` → `(kf_sdpa x, w, 0.25, w, dim=-1, cap=False)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_sdpa
- beat: yes — 1 case(s) run
  - `kernel_sdpa`: cert yes, verify yes, fired 1, in-deriv yes, cost 12 → 1 vs floor 2 → beat yes

### `kern_fuse_sdpa_cap` — review

- instance (corpus:kernel_sdpa_cap): `(matmul (div (exp (tanh (mul (matmul x, w), 0.25))), (sum (exp (tanh (mul (matmul x, w), 0.25))), dim=-1, keepdim=True)), w)` → `(kf_sdpa x, w, 0.25, w, dim=-1, cap=True)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_sdpa_cap
- beat: yes — 1 case(s) run
  - `kernel_sdpa_cap`: cert yes, verify yes, fired 1, in-deriv yes, cost 16 → 1 vs floor 4 → beat yes

### `kern_fuse_layernorm` — review

- instance (corpus:kernel_layernorm): `(mul (sub x, (mean x, dim=-1, keepdim=True)), (rsqrt (add (mean (pow (sub x, (mean x, dim=-1, keepdim=True)), 2), dim=-1, keepdim=True), 1e-05)))` → `(kf_layernorm x, dim=-1, eps=1e-05)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_layernorm
- beat: yes — 1 case(s) run
  - `kernel_layernorm`: cert yes, verify yes, fired 1, in-deriv yes, cost 13 → 1 vs floor 3 → beat yes

### `kern_fuse_rmsnorm` — review

- instance (corpus:kernel_layernorm): `(mul (sub x, (mean x, dim=-1, keepdim=True)), (rsqrt (add (mean (pow (sub x, (mean x, dim=-1, keepdim=True)), 2), dim=-1, keepdim=True), 1e-05)))` → `(kf_rmsnorm (sub x, (mean x, dim=-1, keepdim=True)), dim=-1, eps=1e-05)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_rmsnorm
- beat: yes — 2 case(s) run
  - `kernel_rmsnorm`: cert yes, verify yes, fired 1, in-deriv yes, cost 9 → 1 vs floor 6 → beat yes
  - `kernel_layernorm`: cert yes, verify yes, fired 1, in-deriv no, cost 13 → 1 vs floor 1 → beat no

### `kern_fuse_mm_bias` — review

- instance (corpus:kernel_mm_bias): `(add (matmul x, w), b)` → `(kf_mm_bias x, w, b)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_mm_bias
- beat: yes — 4 case(s) run
  - `kernel_mm_bias`: cert yes, verify yes, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes
  - `kernel_decayed_pair`: cert yes, verify yes, fired 1, in-deriv no, cost 12 → 11 vs floor 11 → beat no
  - `kernel_decayed_chunks`: cert yes, verify yes, fired 1, in-deriv no, cost 11 → 9 vs floor 9 → beat no
  - `kernel_chunk_scan`: cert yes, verify yes, fired 1, in-deriv no, cost 20 → 15 vs floor 15 → beat no

### `kern_fuse_mm_scale` — review

- instance (corpus:kernel_mm_scale_bias): `(mul (matmul x, w), 0.5)` → `(kf_mm_scale x, w, 0.5)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_mm_scale
- beat: yes — 4 case(s) run
  - `kernel_mm_scale`: cert yes, verify yes, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes
  - `kernel_sdpa`: cert yes, verify yes, fired 1, in-deriv no, cost 12 → 1 vs floor 1 → beat no
  - `kernel_sdpa_cap`: cert yes, verify yes, fired 1, in-deriv no, cost 16 → 1 vs floor 1 → beat no
  - `kernel_mm_scale_bias`: cert yes, verify yes, fired 1, in-deriv no, cost 3 → 1 vs floor 1 → beat no

### `kern_fuse_mm_scale_bias` — review

- instance (corpus:kernel_mm_scale_bias): `(add (mul (matmul x, w), 0.5), b)` → `(kf_mm_sb x, w, 0.5, b)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_mm_scale_bias
- beat: yes — 1 case(s) run
  - `kernel_mm_scale_bias`: cert yes, verify yes, fired 1, in-deriv yes, cost 3 → 1 vs floor 2 → beat yes

### `kern_fuse_mm_gelu` — review

- instance (corpus:kernel_mm_gelu): `(gelu (matmul x, w))` → `(kf_mm_gelu x, w)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_mm_gelu
- beat: yes — 1 case(s) run
  - `kernel_mm_gelu`: cert yes, verify yes, fired 1, in-deriv yes, cost 3 → 1 vs floor 3 → beat yes

### `kern_fuse_om_decayed` — review

- instance (corpus:kernel_decayed): `(matmul (mul s, d), v)` → `(om_decayed s, d, v)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_om_decayed
- beat: yes — 8 case(s) run
  - `kernel_decayed`: cert yes, verify yes, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes
  - `kernel_decayed_mask`: cert yes, verify yes, fired 1, in-deriv yes, cost 5 → 4 vs floor 5 → beat yes
  - `kernel_decayed_intra`: cert yes, verify yes, fired 1, in-deriv yes, cost 6 → 5 vs floor 6 → beat yes
  - `kernel_decayed_pair`: cert yes, verify yes, fired 1, in-deriv yes, cost 12 → 11 vs floor 11 → beat no
  - `kernel_decayed_chunks`: cert yes, verify yes, fired 2, in-deriv yes, cost 11 → 9 vs floor 10 → beat yes
  - `kernel_decayed_t512`: cert yes, verify yes, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes
  - `kernel_decayed_t1024`: cert yes, verify yes, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes
  - `kernel_decayed_t2048`: cert yes, verify —, fired 1, in-deriv yes, cost 2 → 1 vs floor 2 → beat yes

### `kern_fuse_chunk_scan` — review

- instance (corpus:kernel_chunk_scan): `(concat (matmul (mul (split sc, sizes=(64, 64), dim=1, index=0), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=0)), (add (matmul (mul (split sc, sizes=(64, 64), dim=1, index=1), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=1)), (mul (unsqueeze (decay_vec (sigmoid gc), t=64), dim=-1), (div (matmul (split sc, sizes=(64, 64), dim=1, index=1), (matmul (split sc, sizes=(64, 64), dim=1, index=0), (split vc, sizes=(64, 64), dim=1, index=0))), 2))), dim=-2)` → `(om_chunk_scan (matmul (mul (split sc, sizes=(64, 64), dim=1, index=0), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=0)), (add (matmul (mul (split sc, sizes=(64, 64), dim=1, index=1), (decay (sigmoid gc), t=64)), (split vc, sizes=(64, 64), dim=1, index=1)), (mul (unsqueeze (decay_vec (sigmoid gc), t=64), dim=-1), (div (matmul (split sc, sizes=(64, 64), dim=1, index=1), (matmul (split sc, sizes=(64, 64), dim=1, index=0), (split vc, sizes=(64, 64), dim=1, index=0))), 2))), dim=-2)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_fuse_chunk_scan
- beat: yes — 1 case(s) run
  - `kernel_chunk_scan`: cert yes, verify yes, fired 1, in-deriv yes, cost 20 → 15 vs floor 16 → beat yes

### `kern_fuse_chunk_scan_3` — unmeasured

- instance (corpus:xcorpus:scan_3): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: no — resource-bound refusal; budget-suspended rules: kern_fuse_addmul, kern_fuse_addmul2, kern_fuse_mul3, kern_fuse_om_decayed, assoc_mul
- siblings excluded: kern_fuse_chunk_scan_3
- beat: — — no registered case exercises this law

### `kern_fuse_chunk_scan_4` — unmeasured

- instance (corpus:xcorpus:scan_4): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: no — resource-bound refusal; budget-suspended rules: kern_fuse_addmul, kern_fuse_addmul2, kern_fuse_mul3, kern_fuse_mm_bias, kern_fuse_mm_scale, kern_fuse_mm_scale_bias, kern_fuse_om_decayed, kern_div_recip, factor_matmul, weight_factor_matmul, comm_add, comm_mul, assoc_add, assoc_mul, aff_lift, affd_lift, affd_lift_post, affd_lift_step, affd_lift_unit, affd_lift_unit_step, affd_lift_unit_step_post, affd_unlift
- siblings excluded: kern_fuse_chunk_scan_4
- beat: — — no registered case exercises this law

### `kern_fuse_chunk_scan_5` — unmeasured

- instance (corpus:xcorpus:scan_5): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: no — resource-bound refusal; budget-suspended rules: kern_fuse_addmul, kern_fuse_addmul2, kern_fuse_mm_bias, kern_fuse_mm_scale, kern_fuse_mm_scale_bias, kern_fuse_om_decayed, kern_div_recip, factor_matmul, comm_add, comm_mul, assoc_add, assoc_mul, naturality_scalar_left, aff_lift, affd_lift, affd_lift_swap, affd_lift_post, affd_lift_step, affd_lift_step_swap, affd_lift_step_post, affd_lift_step_post_swap, affd_lift_unit_step, affd_lift_unit_step_post, affd_unlift, affd_assoc_rev
- siblings excluded: kern_fuse_chunk_scan_5
- beat: — — no registered case exercises this law

### `kern_fuse_chunk_scan_6` — unmeasured

- instance (corpus:xcorpus:scan_6): `(concat (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), (add (add (add (add (add (div (om_decayed s5, d5, v5), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a3, v3)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a4, v4)), 0.5))), dim=-2)` → `(om_chunk_scan (div (om_decayed s0, d0, v0), 0.5), (add (div (om_decayed s1, d1, v1), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q1, (matmul a0, v0)), 0.5))), (add (add (div (om_decayed s2, d2, v2), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q2, (matmul a1, v1)), 0.5))), (add (add (add (div (om_decayed s3, d3, v3), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q3, (matmul a2, v2)), 0.5))), (add (add (add (add (div (om_decayed s4, d4, v4), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q4, (matmul a3, v3)), 0.5))), (add (add (add (add (add (div (om_decayed s5, d5, v5), 0.5), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a0, v0)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a1, v1)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a2, v2)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a3, v3)), 0.5))), (mul (unsqueeze (decay_vec g, t=8), dim=-1), (div (matmul q5, (matmul a4, v4)), 0.5))), dim=-2)`
- truth: equal · cert: yes
- derivable: no — resource-bound refusal; budget-suspended rules: kern_fuse_addmul, kern_fuse_addmul2, kern_fuse_mul3, kern_fuse_mm_bias, kern_fuse_mm_scale, kern_fuse_mm_scale_bias, kern_fuse_om_decayed, kern_div_recip, factor_matmul, weight_factor_matmul, comm_add, comm_mul, assoc_add, assoc_mul, naturality_scalar_left, aff_lift, affd_lift, affd_lift_swap, affd_lift_post, affd_lift_post_swap, affd_lift_step, affd_lift_step_swap, affd_lift_step_post, affd_lift_step_post_swap, affd_lift_unit, affd_lift_unit_post, affd_lift_unit_step, affd_lift_unit_step_post, affd_unlift, affd_assoc_rev
- siblings excluded: kern_fuse_chunk_scan_6
- beat: — — no registered case exercises this law

### `kern_square_strength` — known

- instance (corpus:kernel_layernorm): `(pow (sub x, (mean x, dim=-1, keepdim=True)), 2)` → `(mul (sub x, (mean x, dim=-1, keepdim=True)), (sub x, (mean x, dim=-1, keepdim=True)))`
- truth: equal · cert: yes
- derivable: yes via pow_to_square, square_expand — fixed_point
- siblings excluded: kern_square_strength
- beat: — — —

### `kern_div_recip` — review

- instance (corpus:kernel_strength): `(div y, 4.0)` → `(mul y, 0.25)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: kern_div_recip
- beat: yes — 7 case(s) run
  - `kernel_strength`: cert yes, verify yes, fired 1, in-deriv yes, cost 7 → 1 vs floor 3 → beat yes
  - `kernel_softsign`: cert yes, verify yes, fired 0, in-deriv no, cost 4 → 1 vs floor 1 → beat no
  - `kernel_softmax`: cert yes, verify yes, fired 0, in-deriv no, cost 7 → 1 vs floor 1 → beat no
  - `kernel_softmax_mm`: cert yes, verify yes, fired 0, in-deriv no, cost 14 → 4 vs floor 4 → beat no
  - `kernel_sdpa`: cert yes, verify yes, fired 0, in-deriv no, cost 12 → 1 vs floor 1 → beat no
  - `kernel_sdpa_cap`: cert yes, verify yes, fired 0, in-deriv no, cost 16 → 1 vs floor 1 → beat no
  - `kernel_chunk_scan`: cert yes, verify yes, fired 1, in-deriv yes, cost 20 → 15 vs floor 17 → beat yes

