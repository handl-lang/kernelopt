# Kernel vocabulary — closing the sink's op-table gap

> Followed up by `docs/results_chunk_scan.md` — the schedule half of
> the residual this doc names (`om_chunk_scan` + `kern_fuse_chunk_scan`).

Follow-up to `docs/results_kernel_check.md` §4, which found the
extracted retnet carrier structurally unlowerable through
`KernelSink`: 8 of its 14 ops were outside `supported_ops`
(`decay`, `decay_vec`, `om_decayed`, `split`, `concat`, `linear`,
`contiguous`, `detach_`) and only `kern_div_recip` fired on the
term.  This pass widened the sink table with the missing vocabulary
and re-asked the question: does the extracted carrier lower now,
and how fast?

Same box: RTX 2050 / 16-core / 11 GB.  Parent env torch 2.14.0+cu130
for the retnet arm (the audit's environment); kernelopt venv torch
2.14.1+cu130 / triton 3.8.0 for the bench table.  `fla-core` absent
— the ~0.28 ms FLA reference is the carried retro number, as before.

## What was added

**Binding table** (`sink.py`): nine new entries — `linear`
(`F.linear`, the GEMM+epilogue slot), `concat` (`torch.cat` — a
materialising copy, priced a launch), `split` (a view, zero
launches), `contiguous`/`detach`/`detach_` (views/no-ops, zero
launches), `decay`/`decay_vec` (the decay-mask generators, vendored
carrier semantics), `om_decayed` (the fused `(s⊙d)@v` composite).

**Law bank** (`laws.py`): `kern_fuse_om_decayed` — the 17th law —
folds `matmul(mul(s, d), v)` into `om_decayed(s, d, v)`.  The op
keeps the carrier name: an extracted `om_decayed` enode and this
minted one are the same kernel, compiled through the fused-op path.

**Cost model** (`cost.py`): `_VIEW_OPS` — split/select/reshape-
family/detach/contiguous price at 0 launches (metadata-only on
device).  `concat` deliberately stays a launch.

**Executor** (`sink.py`): the static-subtree memo — a subtree with
no `Var` leaf is input-independent, evaluated once per executor.
The `decay`/`decay_vec` generators and any view-only slices leave
the steady-state loop entirely (the carrier executor's own `_const`
schedule choice).

## Per-op verify + the measured case row

`bench/drivers/probes/kernel_vocab_probe.py`: every new binding
verifies `max_abs=0.0` against its torch reference, and `om_decayed`
verifies against its spelled `matmul(mul(s,d),v)` site (verify-first).

The 17th bench case (`kernel_decayed`, `s(16,128,128) ⊙ d(128,128) @
v`):

| arm | µs | verified |
|---|---|---|
| spelled `matmul(mul(s,d),v)` | 130.3 | — |
| scheduled `om_decayed` (compiled composite) | 96.9 | True (max_abs 0.0) |
| hand-Triton `(s⊙d)@v` | 84.7 | True |

Cost 2→1 launches; Inductor's fused composite is within 1.14× of the
hand kernel — the same signature as the `kf_*` epilogue rows.

## End-to-end: the extracted retnet carrier

T=512/K=4, the audit's own `_saturate` (vendored; root `concat`, 14
distinct ops, 145 nodes):

| check | before | now |
|---|---|---|
| ops outside `supported_ops` | 8/14 | **0/14** |
| lowers | no | **yes — verify32 rel 1.61e-07, passed** |
| kernel laws firing | `kern_div_recip` only | `kern_div_recip` + the `om_decayed` sites are already fused ops |

| arm | med ms |
|---|---|
| spelled-eager32 | 0.391 |
| carrier(gen)32 | 2.557 |
| fused(exec)32 | 2.479 |
| kernel_exec (fused ops compiled) | 5.493 |
| kernel_exec (eager) | 4.293 |
| kernel_exec + manual CUDA graph | **1.011** |
| carrier32 + manual CUDA graph | 0.537 |
| kernel_exec + whole-graph torch.compile | 3.812 |

## The honest gap that remains

The vocabulary gap is closed — the carrier lowers and verifies.
The *speed* gap is **schedule, not vocabulary**: the sink's executor
is a per-op dispatch tape.  `kernel_exec+graph` at 1.011 ms vs
`carrier32+graph(man)` at 0.537 ms isolates it exactly — replayed
per-op launches cost ~2× the pairwise-batched chunk schedule
(`ChunkedDecayModule`, which batches the 25 per-chunk matmuls into a
few GEMMs).  Two further honest notes:

- Compiled `om_decayed` is *slower* per call than its eager composite
  (5.49 vs 4.29 ms) — per-op `torch.compile` dispatch overhead beats
  the fusion win at this op count.  Compilation pays on the bench
  cases (one fused op per program); on a 145-node tape it does not.
- Whole-graph `torch.compile` of the executor (3.81 ms) does not
  recover the schedule — the recursive evaluator graph-breaks per
  binding call; inductor sees fragments, not the term.

The FLA-class residual (~1.6×) needs what the audit already named: a
fused chunk-*scan* schedule (the batched carrier executor's pairwise
batching, or a real scan kernel over the `concat`-of-chunks
structure).  `om_decayed` as a single op cannot express the scan —
that's a schedule-level lowering, the same conclusion, now with the
vocabulary in place to build it on.
