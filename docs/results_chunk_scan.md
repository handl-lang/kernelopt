# Chunked-scan fold — the fused schedule attacking the FLA residual

Follow-up to `docs/results_kernel_vocab.md` (commit `370aa53`): the
vocabulary gap was closed (the retnet carrier lowers+verifies through
`KernelSink`, 0/14 ops outside `supported_ops`) but the speed gap was
named **schedule, not vocabulary** — the per-op dispatch tape + manual
CUDA graph landed 1.011 ms vs the "batched" `ChunkedDecayModule`
carrier's 0.537 ms vs FLA ~0.28 ms (carried retro — `fla-core` absent,
same caveat).  `om_decayed` alone only expresses `(s⊙d)@v` per chunk.

This pass ships the missing piece: `om_chunk_scan`, a schedule-level
op minted by the `kern_fuse_chunk_scan` family — a `concat`-of-chunks
site folds to ONE scheduled block the executor lowers to the
pairwise-batched chunk schedule (`ChunkedDecayModule`'s own plan,
vendored carrier-free into `mlopt_kernel.scan`).

Same box: RTX 2050 / 16-core / 11 GB.  Parent env torch 2.14.0+cu130
for the retnet arm.

## What was added

**`om_chunk_scan`** (`SCHEDULED_KERNEL_OPS`) — a schedule-boundary
marker, not a compiled composite.  Binding-table body is the spelled
`cat` of its flattened arms (the verify-first / `compile_fused=False`
path, and the honest fallback when a node doesn't re-parse as a scan).

**`kern_fuse_chunk_scan` + `kern_fuse_chunk_scan_{3..6}`**
(`laws.py`) — `concat(x₁..xₙ, dim) → om_chunk_scan(x₁..xₙ, dim)`.
Concat is variadic in the IR while patterns are fixed-arity, so the
law is a family (the `concat_binarize_{n}` precedent): the binary
member covers the extracted carrier's nested-binary concat spine —
each minted `om_chunk_scan` flattens again at the level above until
the whole spine is marked; the n-ary members cover flat spellings.
The `check` runs the real plan parse
(`mlopt_kernel.scan.is_chunk_scan`): every flattened arm must carry
exactly one decayed intra element — `om_decayed(s,d,v)` *or* its
spelled `matmul(mul(s,d),v)` site — plus decay-gated pair products
`row ⊙ (q @ (A @ v)) / S`, under uniform concrete chunk shapes.  A
concat that fails the parse keeps its spelled evaluation.

**`mlopt_kernel.scan`** — the vendored schedule: arm parser
(`_peel_summand`/`_classify_body`/`_parse_arm`), slice restack
(`_sliced_gather`/`_contig_chunk_range` — split/select stacks become
reshape+permute *views*, zero kernels), plan assembly
(uniform-shape gate, q/state dedup tables, pair indices), and
`ScanRunner` — the pairwise-batched forward: batched score GEMM +
decay-product + `@v` contraction for the intra half; one batched
`A_j @ v_j` state GEMM + one gathered `q_i @ M_j` GEMM + segment-sum
accumulation for the cross half.  Operand evaluation goes through
the executor's `ev(t)` channel: per-forward memo for live subterms,
static memo for the `decay`/`decay_vec` gates.

**Executor** (`sink.py`) — two changes:

- *scheduled dispatch*: `om_chunk_scan` nodes route to a cached
  `ScanRunner` under `compile_fused`; a node whose arms no longer
  parse falls back to the spelled `cat` — never a silent drop.
- *per-forward memo*: live op results cache for the call's duration
  (the carrier's own memoised-eval choice) — a DAG-shared subterm
  (the carrier's shared q/k/v projections feeding every chunk's
  `split`) evaluates once per forward, not once per use.  This pays
  the *spelled tape* too: `kernel_tape+graph` moved 1.011 → 0.537 ms.

**Cost** (`cost.py`) — `om_chunk_scan` prices at **0 launches**: the
marker emits no kernel of its own — the block's GEMMs are billed at
its arm operands.  Below the `concat` it replaces (1 launch), so a
marked concat spine strictly wins per node and the nested fold
extracts whole rather than piecemeal (alphabetical member order would
otherwise hand ties to `concat`).

## The fold on the real extraction

T=512/K=4 `_saturate` (the audit's config, vendored): root `concat`,
14 distinct ops, 145 nodes — extracted as a **nested binary** concat
over **7 chunk arms of mixed size**: 4 chunks of 32 + 3 of 128.

- `kern_fuse_chunk_scan` mints `om_chunk_scan` at **5 of 6** concat
  nodes (+5 enodes); extraction picks each — `concat(om_chunk_scan,
  om_chunk_scan)` at the root.
- The root concat does **not** fold: the uniform-shape gate vetoes
  the mixed 32/128 group — the same gate `ChunkedDecayModule` has
  (`build_decayed_plan` returns `False` on this exact term — verified
  below).  What remains spelled is one `cat` of two scheduled blocks.
- `kernel_cost` 206 → 201.

## The measured ladder (fp32, CUDA, median of 30)

| arm | med ms | note |
|---|---|---|
| spelled-eager32 | 0.396 | the module — one big unchunked masked mm |
| fused(exec)32 | 2.634 | incumbent per-op executor, eager |
| kernel_tape (per-op dispatch) | 3.012 | memo'd; was 4.29 eager / 5.49 compiled |
| kernel_scan (scheduled, eager) | 2.154 | batched plan, python-dispatched |
| kernel_scan (spelled-cat path) | 2.262 | `compile_fused=False` fallback |
| kernel_tape + manual CUDA graph | 0.537 | was **1.011** — the memo alone |
| carrier32 + manual CUDA graph | 0.538 | **serial fallback** — see below |
| fused32 + CUDA graph (incumbent) | 0.454 | audit's delivery path (carried) |
| **kernel_scan + manual CUDA graph** | **0.271** | two scheduled blocks + one cat |
| fla-chunk | ~0.28 | carried retro — `fla-core` absent |

verify32: scan rel **1.93e-07 passed** vs the spelled fp32 module;
spelled-cat path rel 1.61e-07; tape rel 1.61e-07; carrier rel
1.61e-07.

## What the ladder actually says

**The headline finding is honest in both directions:**

- `kernel_scan+graph` at **0.271 ms beats every arm** — the carrier
  (0.537/0.538), the incumbent fused32+graph delivery (0.454), even
  the spelled eager module (0.396) — and lands *at or below* the
  ~0.28 ms FLA retro reference.  The named residual is attacked.
- The 0.537 "batched carrier" arm turns out to be **serial**:
  `build_decayed_plan` returns `False` on this extraction — the
  4×32 + 3×128 mixed chunk sizes fail its uniform-shape gate, so the
  0.537 was the serial IRModule under a graph all along.  With the
  new per-forward memo our own tape+graph reaches the same 0.537 —
  they are the same schedule.  The pairwise-batched carrier never ran
  on this term; `om_chunk_scan` is the **first** lowering that
  actually batches it (per uniform-chunk group).
- Eagerly, the scheduled block still pays Python-level plan replay
  (2.15 ms) — the win is a *schedule* win, realised under graph
  replay (or amortised by construction at carrier scale).  The small
  bench row `kernel_chunk_scan` (2 chunks, d=64) honestly reports
  0.198× — Python-bound at toy sizes, same signature as
  `kernel_decayed_intra`'s 0.546×.

## The honest residual

- **Mixed chunk sizes stay split.**  A non-uniform arm set folds per
  uniform group — here two scheduled blocks under one spelled `cat`.
  A whole-scan fold would need per-group plans under one marker;
  same math, one less launch — measured headroom ~1 cat.
- **The FLA comparison is at parity on a launch-bound term.**  At
  T=512/d=16 the scheduled block still emits ~10 kernels; FLA's real
  kernel does the scan in-register.  At larger T/K the GEMM-bound
  regime is where the hand kernel would pull away — the honest next
  arm is a real fused Triton scan in `bench/limits/triton/` style
  (option (b) of the task — not built; the batched-composite route
  was the architecture the sink already supports).
- **`torch.compile` of the scheduled block** is unexplored — the
  runner is a term-walking evaluator, not a compilable composite;
  making the plan tape compile-friendly is future work.

`bench/drivers/probes/kernel_scan_probe.py` reproduces the ladder
(parent env, `PYTHONPATH=<kernelopt>/src`); the `kernel_chunk_scan`
row lives in `python -m bench.discovery.kernel`.
