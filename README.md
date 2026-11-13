# kernelopt — **private**

The codegen domain for the mlopt game layer, split out of
[`handl-lang/graphopt`](https://github.com/handl-lang/graphopt)
(plan 0032 phase 4).  **This repo is private: nothing here may be
referenced from graphopt's public docs or the paper.**

Contents:

- `src/mlopt_kernel/` — the domain package:
  - `laws.py` — the codegen law bank (`KERNEL_RULES`: named
    fused-kernel ops + arithmetic strength reductions),
  - `cost.py` — `kernel_cost` (launch/weight-dominated),
  - `sink.py` — `KernelSink`/`KernelExecutor` (the
    `mlopt_core.ports.Sink` port over a torch composite —
    verify-first, the ONNX precedent),
  - `board.py` — `KernelBoard` (the `gamekit.engine.Board` episode;
    `stop` is a scored move),
  - `weights.py` — `KERNEL_PLAYER_WEIGHTS`, the shipped learned prior,
  - `priorart.py` — `KERNEL_PRIOR_ART_RULES`, the kernel-compilers'
    known codegen vocabulary as laws (the `pa_kf_*` expand bank the
    novelty sieve derives against),
  - `scan.py` — the `om_chunk_scan` schedule: `build_scan_plan` /
    `is_chunk_scan` term analysis + `ScanRunner`, the pairwise-batched
    forward the `kern_fuse_chunk_scan*` laws lower to,
  - `search.py` — the ADR-0165 kernel-IR term search (gate → bound →
    measure; `lib-kernel-ir` `Term`s in, routed `Candidate`s out),
  - `play.py` — the **`kernel` domain plugin**: importing it seats
    the domain into `mlopt_discovery.play.DOMAINS` and its players
    into the gamekit registry — run
    `python -m mlopt_kernel.play --domain kernel --deliver`.
- `tests/` — `test_kernel.py` (laws, cost, sink, board, play domain)
  plus `test_kernel_priorart.py` / `test_kernel_sieve.py` /
  `test_kernel_search_ir.py` / `test_overnight.py` / `test_transfer.py`.
- `scripts/train_kernel_prior.py` — retrain + emit the baked prior.
- `bench/` — the measured half: `bench/discovery/kernel.py` (the 46
  kernel cases — `python -m bench.discovery.kernel` from the repo
  root), `bench/limits/triton/` (the hand-Triton field baseline;
  needs `triton` + CUDA, declines honestly without), and
  `bench/drivers/` (the overnight loop, the sieve/prior-art screens,
  the probes, and the transfer driver — result documents, not
  shipped code).
- `docs/results_kernel_check.md`, `notes/kernel-engineer-board.md` —
  the audit + design notes moved out of the parent repo.

## Wiring

The substrate is consumed as **git+ssh deps on per-package
subdirectories** of the split sibling repos, pinned to `main` for
now — see `[tool.uv.sources]` in `pyproject.toml`:

| dep | repo | source |
|---|---|---|
| `mlopt-core` | `handl-lang/eqsat` | `packages/mlopt-core` |
| `mlopt-orchestrator` | `handl-lang/eqsat` | `packages/mlopt-orchestrator` |
| `mlopt-torch` | `handl-lang/graphopt` | `packages/mlopt-torch` |
| `mlopt-carriers` | `handl-lang/graphopt` | `packages/mlopt-carriers` |
| `mlopt-discovery` | `handl-lang/graphopt` | `packages/mlopt-discovery` |
| `gamekit` | `handl-lang/gamekit` | `packages/gamekit` |
| `benchkit` | `handl-lang/gamekit` | `packages/benchkit` |
| `lib-kernel-ir` | sibling checkout | `../lib-kernel-ir` (path dep) |

## Verify

```sh
uv sync                                  # resolves the git deps
.venv/bin/python -m pytest tests/ -q     # the kernel suite
.venv/bin/python -m mlopt_kernel.play --domain kernel \
    --train-cases 8 --eval-cases 4 --budget 8
```
