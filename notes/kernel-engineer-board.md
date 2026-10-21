# Note — the kernel-engineer board (reusing the game layer)

**PRIVATE. 2026-10-20: spike verdict GO — both criteria passed**
(commit `3801504`, `packages/mlopt-kernel/`): verified-equal kernels at
lower measured cost (silu 2.24×, softsign 3.13×, FMA 1.36× on CUDA
fp32) and the trained player beat playbook/random on kernel boards
(0.49 / 0.34 / 0.08, 24 held-out cases).  Stays private until it earns
publication — absent from org docs, pitch, paper.

## 2026-10-20 (later): expansion arm — measured results

Bank grew 5 → 10 laws (the Inductor-pattern arm: deeper fusions +
epilogues, `cond`/`dspec` side conditions as data like the default
bank's).  All verify-first — every fused op passes `sink.verify`
against its spelled site (fp64 feeds) and the in-graph certificate.

New laws (each an extraction win under `kernel_cost`, verified in
`tests/test_kernel.py`):

| law | site → fused op | guard |
|---|---|---|
| `kern_fuse_swiglu` | `(a·σ(a))·b` → `kf_swiglu` | — (deeper than `mul(kf_silu,·)`) |
| `kern_fuse_softmax` | `exp(u)/Σexp(u)` → `kf_softmax` | keepdim + single-axis (mirrors `softmax_fold`) |
| `kern_fuse_layernorm` | `c·rsqrt(mean(c²)+eps)`, `c = x−mean(x)` → `kf_layernorm` | keepdim, numeric eps, pow 2 |
| `kern_fuse_mm_bias` | `mm(a,w)+b` → `kf_mm_bias` | — |
| `kern_fuse_mm_scale_bias` | `mm(a,w)·s+b` → `kf_mm_sb` | — (beats `kf_addmul` two-launch) |

Case table (`bench.discovery.kernel`, 11 cases, CUDA fp32, ~50 reps,
median).  **Every row verified, fused ops included.**

| case | cost | spelled µs | sched µs | speedup | triton µs (verified) |
|---|---|---|---|---|---|
| kernel_silu | 3→1 | 476 | 209 | 2.28× | 200 ✓ |
| kernel_softsign | 4→1 | 665 | 209 | 3.19× | 201 ✓ |
| kernel_fma | 4→3 | 756 | 563 | 1.34× | 291 ✓ |
| kernel_swiglu | 4→1 | 755 | 302 | 2.50× | 292 ✓ |
| kernel_softmax | 7→1 | 668 | 208 | 3.21× | 199 ✓ |
| kernel_layernorm | 13→1 | 1056 | 209 | 5.05× | 200 ✓ |
| kernel_mm_bias | 2→1 | 899 | 731 | 1.23× | 900 ✓ |
| kernel_mm_scale_bias | 3→1 | 1114 | 990 | 1.13× | 904 ✓ |
| kernel_strength | 7→3 | 662 | 661 | 1.00× | 290 ✓ |
| kernel_mixed | 6→2 | 940 | 487 | 1.93× | 290 ✓ |
| kernel_flat (control) | 4→4 | 661 | 661 | 1.00× | — skip (no kernel) |

**The honest triton gap.**  Hand Triton reaches ~200–290 µs on every
pointwise/row case — the compiled `kf_*` composite matches it on the
single-fusion cases (silu 209, softsign 209, softmax 208, layernorm
209 ≈ triton 199–200), but cases the law bank fuses *partially* show
the residue: `fma` (addmul fused, `tanh` still a launch) 563 vs 291,
`mixed` (silu fused, outer `σ` left) 487 vs 290, `strength` (two
reductions, no fusion law) 661 vs 290.  The mm epilogues invert it:
Inductor's mm path (731/990) beats the tutorial-tile hand kernel
(900/904) at this size.  Net: **the codegen gap is real but now
visible and bounded — it is the leftover-pointwise fusions, not the
fused ops themselves.**  A second-generation "fuse the surviving
pointwise args into the kf op" law family is the measured next arm.

Player deep-dive (`bench.players.run --boards kernel`, 24 held-out,
budget 8 — the leaderboard protocol, arms re-registered unchanged):

| player | reward/ep | evals/ep | note |
|---|---|---|---|
| `linear-greedy` | **0.560** | 2.29 | short lines — extracts early |
| `linear` | 0.499 | 3.42 | |
| `playbook` | 0.200 | 8.0 | drains the budget |
| `linear-cold` | 0.145 | 6.17 | |
| `random` | 0.099 | 5.63 | |
| `distilled` | — | — | does not play this board (contraction artifact) |
| `neural` | — | — | does not play this board (open-board only) |

Verdicts on the four expansion arms:

1. **More laws — PASS.**  All five verify numerically and pay on
   measured CUDA time (1.13–5.05×); the layernorm fold is the biggest
   win (13→1 launches, 5.05×).  Guards hold (keepdim=False,
   non-numeric eps, non-square pow all decline).
2. **Triton adapter — PASS.**  Real hand kernels for all 10
   non-control cases (incl. tiled mm+bias/scale-bias); verified-equal
   every case they cover; honest skip on `kernel_flat`.
3. **More cases — PASS.**  5 → 11 incl. the no-site `kernel_flat`
   control (cost 4→4, ~1.00× — honest negative).
4. **Player deep-dive — PASS with a caveat.**  The shipped
   `KERNEL_PLAYER_WEIGHTS` (trained on the 5-law bank) generalizes:
   0.56 greedy on the expanded bank, still > playbook 0.20 — the
   `x:m` match-oracle feature carries the new laws untrained
   (`a:r:*` buckets for the new rules are zero-weight).  Caveat:
   cold-start REINFORCE on the expanded bank plateaus at ~0.20–0.28
   at 600 cases — the imitation phase is what carries the prior;
   retraining a v2 prior is the open arm.

The earlier verdict stands. 

## 2026-10-21: real-model + gen-2 arm — measured results

Bank grew 10 → 16 laws, in two families:

* **real-model patterns** — `kern_fuse_mm_scale` (`mm·s` → `kf_mm_scale`,
  the bias-free `QKᵀ·(1/√d)` spelling), `kern_fuse_softmax_mm`
  (`softmax(u)·v` → `kf_softmax_mm`, the post-softmax value matmul —
  same keepdim/single-axis cond as the softmax fold),
  `kern_fuse_rmsnorm` (gain-free `x·rsqrt(mean(x²)+eps)` → `kf_rmsnorm`,
  the kernel twin of `rms_norm_fold_nogain`'s site), `kern_fuse_mm_gelu`
  (`gelu(mm)` → `kf_mm_gelu`, the transformer-MLP epilogue).
* **gen-2 residue folds** — the measured triton gap named these:
  `kern_fuse_addmul_tanh` (`a·b+tanh(c)` → `kf_addmul_tanh`, the tanh
  `kernel_fma` left spelled) and `kern_fuse_silu_sigmoid`
  (`(a·σa)·σb` → `kf_silu_sigmoid`, the σ `kernel_mixed` left).  Plus
  a composition note, not a new law: the strength-reduced pair exposes
  an `addmul` site (`x·x + y·¼` → `kf_addmul`), and the sdpa chain
  extracts `kf_softmax_mm(kf_mm_scale(q,k,s), v)` — 2 launches from a
  12-cost spelled attention block.

Case table (16 cases, CUDA fp32, ~50 reps, median).  Every row
verified, fused ops included.

| case | cost | spelled µs | sched µs | speedup | triton µs (verified) |
|---|---|---|---|---|---|
| kernel_silu | 3→1 | 476 | 213 | 2.23× | 202 ✓ |
| kernel_softsign | 4→1 | 682 | 212 | 3.21× | 203 ✓ |
| kernel_fma | 4→1 | 772 | 309 | **2.50×** | 338 ✓ |
| kernel_swiglu | 4→1 | 779 | 305 | 2.55× | 294 ✓ |
| kernel_softmax | 7→1 | 684 | 213 | 3.21× | 203 ✓ |
| kernel_layernorm | 13→1 | 1057 | 214 | 4.93× | 202 ✓ |
| kernel_mm_bias | 2→1 | 1011 | 696 | 1.45× | 912 ✓ |
| kernel_mm_scale_bias | 3→1 | 1053 | 932 | 1.13× | 905 ✓ |
| kernel_mm_scale | 2→1 | 937 | 929 | **1.01× — parity** | 944 ✓ |
| kernel_mm_gelu | 3→1 | 901 | 957 | **0.94× — slight loss** | 941 ✓ |
| kernel_rmsnorm | 9→1 | 507 | 213 | 2.38× | 201 ✓ |
| kernel_softmax_mm | 14→4 | 3561 | 2065 | 1.72× | 2064 ✓ |
| kernel_sdpa | 12→2 | 3580 | 2137 | 1.68× | 2087 ✓ |
| kernel_strength | 7→2 | 665 | 480 | **1.39×** | 294 ✓ |
| kernel_mixed | 6→1 | 942 | 308 | **3.06×** | 294 ✓ |
| kernel_flat (control) | 4→4 | 664 | 663 | 1.00× | — skip (no kernel) |

**The residue gap closed where it could.**  The three cases the last
arm showed surviving pointwise args on — `fma` (563→309 µs, now
≈/beats triton), `mixed` (487→308, ≈ triton), `strength` (661→480,
still 2 launches vs triton's 1 — the honest remainder) — all moved to
the one-launch composite.  The mm-epilogue rows stay the inversion:
Inductor's mm path beats the tutorial-tile hand kernel (696/932 vs
912/905), and the two *new* epilogue rows are the arm's honest
negatives — `mm_scale` parity (1.01×) and `mm_gelu` a slight loss
(0.94×): at this size the mm dominates and the absorbed pointwise is
noise — the laws are real (cost 2→1, verify-equal) but don't pay on
measured time here.  The attention chains land between: our 2-launch
composite ties the 3-launch hand pipeline (softmax_mm 2065 vs 2064,
sdpa 2137 vs 2087 — both verified).  **The remaining codegen gap is
now explicitly the un-folded matmuls** (a flash-style
`mm→softmax→mm` single launch) and the `strength` row's second launch
— next arm, if any: a fused-attention `kf_sdpa` op and a
pointwise-pair `kf` op.

**The spelled-overflow finding.**  The first `kernel_softmax_mm`
draft (`softmax(qk)·v`, unscaled) *failed* verify — not a law bug:
the spelled reference's `exp(qk)` overflows fp32 at bench scale
(→ inf, then inf/inf → NaN) where the fused max-shifted softmax
stays finite.  The fused op is strictly more numerically stable than
the spelling it folds — documented here, and the case now carries
the *bounded* soft-cap spelling `softmax(tanh(qk))·v` (a real
pattern — Gemma-2's logit soft-capping) so the timed row compares
finite references.

**Arm 2 — the learned-player gap.**  `scripts/train_kernel_prior.py`
(the `train_meta_prior.py` sibling) retrains the prior on the
16-law bank: match-then-stop imitation + REINFORCE (64 cases × 8 ep).
Held-out (32 cases, seed+7919):

| arm | reward/ep | note |
|---|---|---|
| shipped-greedy | 0.646 | v1 weights (trained on 5 laws) via `x:m` |
| trained-greedy | 0.646 | **v2 — parity, all `a:r` buckets trained** |
| trained | 0.589 | |
| shipped | 0.584 | |
| cold-rl-greedy | 0.287 | the imitation-dependence control |
| scripted | 0.276 | |
| random | −0.045 | |
| cold | −0.106 | |

The v2 table is **baked** (`KERNEL_PLAYER_WEIGHTS`): held-out parity
with v1 but honest provenance — every rule in the current bank has a
trained `a:r:*` bucket (v1's new-law buckets were zero).  Under the
leaderboard protocol (24 held-out, budget 8 — enlarged case pool):
`linear-greedy` 0.652, `linear` 0.587, `playbook` 0.290,
`linear-cold` −0.097, `random` −0.034.

Verdicts on this arm's picks:

1. **Real-model fusion laws — PASS with honest negatives.**  All six
   verify numerically; four pay on measured time (rmsnorm 2.38×,
   softmax_mm 1.72×, sdpa 1.68× + the residue folds), and two mm
   epilogues measure parity/loss (mm_scale 1.01×, mm_gelu 0.94× —
   mm-dominated at this size).  The spelled-overflow finding is a
   bonus: the fused softmax is strictly more stable than the spelling.
2. **The learned-player gap — PARTIAL.**  Retraining on the
   16-law bank gives parity held-out (0.646 = 0.646) with all
   `a:r` buckets trained — the prior no longer leans on `x:m`
   alone, but *the imitation dependence does not shrink*: cold
   REINFORCE still plateaus (~0.29 greedy at 512 episodes vs
   0.646 warm-started).  The `a:r` buckets were not the bottleneck;
   the missing piece is whatever carries cold-start exploration.
   Curriculum/useful-tier variants were not tried in this window.

## 2026-10-21 (later): flash-style `kf_sdpa` + `om_decayed` cases — measured results

Bank grew 17 → 19 laws.  Two folds for the named codegen gap — the
un-folded matmuls around softmax — and four carrier-style measured
cases for the `om_decayed` vocabulary.

New laws (verify-first; same keepdim/single-axis gate + `dim` unwrap
as the softmax site):

| law | site → fused op | note |
|---|---|---|
| `kern_fuse_sdpa` | `matmul(softmax(mm·s), v)` → `kf_sdpa(a,w,s,v, cap=False)` | the whole attention block, one launch |
| `kern_fuse_sdpa_cap` | `matmul(softmax(tanh(mm·s)), v)` → `kf_sdpa(…, cap=True)` | the Gemma-2 soft-cap spelling |

`kernel_sdpa` now extracts `kf_sdpa(x,w,s,v)` — cost **12→1** (was
`kf_softmax_mm(kf_mm_scale)` at 12→2).  New case `kernel_sdpa_cap`
covers the bounded-logit twin (16→1).  New triton arms:
`_triton_sdpa_cap` (mm·s → tanh-softmax → mm pipeline) and
`_triton_decayed_mask` (`tril(σ(g)^(i−j))` per call + the batched
`(s⊙d)@v`).

Case table (22 cases, CUDA fp32, ~50 reps, median — new/changed rows
in **bold**; the rest re-measured this run and drifted within the
documented band):

| case | cost | spelled µs | sched µs | speedup | triton µs (verified) |
|---|---|---|---|---|---|
| kernel_sdpa | **12→1** | 3615 | 2029 | **1.78×** | 2075 ✓ |
| kernel_sdpa_cap (new) | 16→1 | 3911 | 2038 | **1.92×** | 2077 ✓ |
| kernel_decayed | 2→1 | 75 | 91 | **0.83× — honest loss** | 81 ✓ |
| kernel_decayed_mask (new) | 5→4 | 111 | 138 | **0.80× — loss** | 112 ✓ |
| kernel_decayed_intra (new) | 6→5 | 108 | 146 | **0.74× — loss** | — skip |
| kernel_decayed_pair (new) | 12→11 | 198 | 219 | **0.90× — loss** | — skip |
| kernel_decayed_chunks (new) | 11→9 | 653 | 278 | **2.35× (see below)** | — skip |

**Verdicts.**

1. **`kf_sdpa` — PASS, and it flips the triton comparison.**  The
   2-launch chain last arm *trailed* the 3-launch hand pipeline
   (2137 vs 2087); the whole-block fold now *beats* it (2029 vs 2075;
   capped 2038 vs 2077).  Both caps verify-equal and the keepdim
   gate declines honestly.  The un-folded-matmuls gap is closed as
   far as the compiled-composite sink reaches — a real flash/online-
   softmax kernel is a schedule-level lowering, not another fold.
2. **`om_decayed` measured cases — PASS on lowering, honest negative
   on pay.**  All four carrier spellings lower and verify; all
   measure **0.74–0.90×** — the fold removes one launch but adds a
   per-op `torch.compile` dispatch premium that exceeds it at these
   sizes.  A scale probe on `(s⊙d)@v` (16·128² → 128·512²) shows the
   deficit shrinks to parity (0.80→0.88→0.99) and never inverts:
   the masked-score materialisation the fold saves is bandwidth-
   cheap next to a real GEMM — unlike the 4096×512 pure-pointwise
   sites where fusion removes the dominant traffic.  `kernel_decayed`
   itself straddles the band (0.80–1.17 across re-runs; the vocab
   doc's 130→97 was a favourable draw).  `kernel_decayed_chunks`'
   2.35× did not reproduce (0.67–0.79 on re-runs) — treat as
   ~parity, variance-dominated.  This is the vocab doc's own caveat
   measured row-by-row: **per-op compile dispatch beats the fusion
   win on mm-adjacent ops** — the launch-counting cost model
   over-prices this fold, and the real residual stays the
   schedule-level chunked batching the audit already named.
   (Case-design note: decay rates are live `sigmoid(g)` feeds — the
   carrier parametrisation; a `Const` rate pins the mask generator
   to CPU and crashes under CUDA feeds.)

Prior caveat stands: the shipped `KERNEL_PLAYER_WEIGHTS` carries the
new laws through `x:m`/`a:r:*` buckets untrained — a v3 prior retrain
remains the open arm.

## 2026-10-21 (latest): pointwise-pair `kf` ops + the `om_decayed` size ladder — measured results

Bank grew 19 → 21 laws.  Two pointwise-pair folds close the residue
shapes the strength reductions expose — the last named codegen gap —
plus a sized `om_decayed` ladder pricing the fold's crossover
directly.

New laws (verify-first; no guards — the sites are shape-free):

| law | site → fused op | note |
|---|---|---|
| `kern_fuse_addmul2` | `add(mul(a,b), mul(c,d))` → `kf_addmul2(a,b,c,d)` | the two-product sum `x·x + y·¼` `kern_fuse_addmul` reached as `kf_addmul(a,b, c·d)` — a fused op plus a surviving spelled mul |
| `kern_fuse_mul3` | `mul(mul(a,b), c)` → `kf_mul3(a,b,c)` | the weighted-square `pow(x,2)·w` after `kern_square_strength` |

Both pay only when the children are leaves — a spelled composite
child stays costed inside the fused op's args, so `kf_swiglu` still
beats `kf_mul3(a, σa, b)` (3 vs 1) and the law never competes with a
deeper fold.  `kernel_strength` now extracts
`kf_addmul2(x, x, y, 0.25)` — the second launch is gone.

Case table (28 cases, CUDA fp32, ~50 reps, median — new/changed rows
in **bold**; unchanged rows re-measured this run, drifting within the
documented band — this host's absolute µs run a few % off the last
table's, e.g. sdpa 1971 vs 2029):

| case | cost | spelled µs | sched µs | speedup | triton µs (verified) |
|---|---|---|---|---|---|
| kernel_strength | **7→1** | 695 | 369 | **1.88×** | 333 ✓ |
| kernel_powscale (new) | 5→1 | 507 | 314 | **1.61×** | — skip (no kernel) |
| kernel_addmul2 (new) | 3→1 | 762 | 312 | **2.44×** | — skip (no kernel) |
| kernel_decayed | 2→1 | 138 | 199 | **0.69× — honest loss** | 119 ✓ |
| kernel_decayed_mask | 5→4 | 122 | 156 | **0.78× — loss** | 127 ✓ |
| kernel_decayed_intra | 6→5 | 141 | 162 | **0.87× — loss** | — skip |
| kernel_decayed_pair | 12→11 | 237 | 650 | **0.36× — loss** | — skip |
| kernel_decayed_chunks | 11→9 | 288 | 1129 | **0.26× — loss** | — skip |
| kernel_decayed_t512 (new) | 2→1 | 1080 | 1113 | **0.97×** | — skip |
| kernel_decayed_t1024 (new) | 2→1 | 3854 | 3939 | **0.98×** | — skip |
| kernel_decayed_t2048 (new) | 2→1 | 14720 | 15002 | **0.98×** | — skip |
| kernel_decayed_t4096 (new) | 2→1 | 60984 | 61600 | **0.99×** | — skip |

**Verdicts.**

1. **Pointwise-pair `kf` ops — PASS.**  Both verify-equal and both
   pay: `kernel_strength` closes its second launch (480→369 µs,
   1.39→1.88×, now ~90% of the hand triton kernel's 333 rather than
   61% of it); `kernel_powscale` (1.61×) and `kernel_addmul2` (2.44×)
   land in the single-fusion band.  The residue gap the last two
   arms named is closed as far as the `kf_*` vocabulary reaches —
   every surviving spelled pointwise op in the case table is now a
   deliberate honest negative, not a missing fold.
2. **`om_decayed` asymptote — measured, honest answer: it never
   pays.**  The ladder is monotone in T — 0.97 → 0.98 → 0.98 →
   0.99 — asymptoting to parity at T=4096 without ever inverting,
   confirming the earlier scale probe (0.80→0.88→0.99) on the timed
   row.  The deficit is the per-op `torch.compile` dispatch premium
   (~30–600 µs across the sweep), and the fold's win — one fewer
   `(…,T,K)` materialisation — stays bandwidth-trivial next to a
   real GEMM at every size measured.  The small-case rows straddle
   0.69–0.87 this run (`decayed_pair`/`decayed_chunks` swung deep on
   recompile-limit churn — two fused ops, two dispatches; treat as
   variance-dominated, consistent with the earlier 0.67–0.79 band).
   **The `om_decayed` fold is a vocabulary win, not a codegen win:**
   it lets the bank *name* the decayed element carriers extract, and
   the launch-counting cost model prices it 1 launch honestly — but
   on this sink the measured pay is ≤ parity at every size.  The
   real residual stays the schedule-level chunked batching the audit
   named.

Named next arms, updated: the `kf_sdpa` variants (causal-mask and
`enable_gqa` spellings — real model patterns the bank cannot yet
name) and the v3 prior retrain.  The `strength`/`mixed`-class
residue story is now complete.

2026-10-19, post-Lotufo.  The same machinery that plays law-discovery
can play kernel engineering — a second domain, not a second system.

## The mapping

| law-discovery game | kernel board |
|---|---|
| candidates = rewrite candidates | candidates = kernel/codegen variants (tile, fuse, vectorize, unroll, schedule) |
| referee = cert + pay gate | referee = numeric verify + measured wall/mem |
| score = reward tiers | score = measured speedup vs torch/Triton baselines |
| observability wall = pending value invisible | same wall: a kernel's pay is invisible until compiled+measured |

## What transfers directly

- `Board` protocol + domain registry (`python -m mlopt_discovery.play
  --domain …`) — a kernel domain registers like `meta|open|contraction`
- players package (`random/playbook/linear/neural/distilled`) +
  leaderboard runner (`bench/players`)
- stop-policy rounds 1–8 (ADR 0013): per-step credit, calibrated value
  prediction, break-even gates, the shared-softmax delivery hazard —
  "is compiling this candidate worth the oracle call" is the same stop
  problem, already characterized
- `bench/discovery` case machinery — kernel variants become another
  case family (category `speed`/`memory`)

## Where it lives

Package-level reuse in this monorepo — a sibling package
(`mlopt-kernel/` next to `mlopt-carriers`/`mlopt-onnx`) holding domain
rules, kernel sinks, fixtures. The engine/game layer untouched — same
pattern as ONNX landing with zero core edits. No repo split: boards
and engine version together.

## Why it matters

Autotuning becomes a learned game under a certified referee — and
open-question #5 ("does the prior transfer to a game that grows under
it?") gets a second domain to answer it. The measured payoff also
closes the codegen gap the honest limits already show (FLA ~11× —
kernel quality, not semantics).
