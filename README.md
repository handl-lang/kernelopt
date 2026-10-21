# kernelopt — **private**

The codegen domain for the mlopt game layer, split out of
[`handl-lang/research-mlopt`](https://github.com/handl-lang/research-mlopt)
(plan 0032 phase 4).  **This repo is private: nothing here may be
referenced from research-mlopt's public docs or the paper.**

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
  - `play.py` — the **`kernel` domain plugin**: importing it seats
    the domain into `mlopt_discovery.play.DOMAINS` and its players
    into the gamekit registry — run
    `python -m mlopt_kernel.play --domain kernel --deliver`.
- `tests/` — `test_kernel.py` (laws, cost, sink, board, play domain).
- `scripts/train_kernel_prior.py` — retrain + emit the baked prior.
- `bench/` — the measured half: `bench/discovery/kernel.py` (the 16
  kernel cases — `python -m bench.discovery.kernel` from the repo
  root) and `bench/limits/triton/` (the hand-Triton field baseline;
  needs `triton` + CUDA, declines honestly without).
- `docs/results_kernel_check.md`, `notes/kernel-engineer-board.md` —
  the audit + design notes moved out of the parent repo.

## Wiring

The substrate is consumed as **git+ssh deps on per-package
subdirectories** of the (public) research-mlopt monorepo, pinned to
`main` for now — see `[tool.uv.sources]` in `pyproject.toml`:

| dep | source |
|---|---|
| `mlopt-core` | `packages/mlopt-core` |
| `mlopt-torch` | `packages/mlopt-torch` (transitive) |
| `mlopt-orchestrator` | `packages/mlopt-orchestrator` (transitive) |
| `gamekit` | `packages/gamekit` |
| `mlopt-discovery` | `packages/mlopt-discovery` |

## Verify

```sh
uv sync                                  # resolves the git deps
.venv/bin/python -m pytest tests/ -q     # the kernel suite
.venv/bin/python -m mlopt_kernel.play --domain kernel \
    --train-cases 8 --eval-cases 4 --budget 8
```
