# kernelopt

The codegen domain for the mlopt game layer.  It seats a `kernel`
board into the same game/eqsat machinery that plays law discovery:
the moves are codegen laws (fused-kernel ops + strength reductions),
the referee verifies numerically *and* on measured wall time.

## Packages

- `src/mlopt_kernel/` — the domain package:
  - `laws.py` — the codegen law bank (`KERNEL_RULES`: 28 named
    fused-kernel ops + arithmetic strength reductions),
  - `cost.py` — `kernel_cost` (launch/weight-dominated),
  - `sink.py` — `KernelSink`/`KernelExecutor` (the
    `mlopt_core.ports.Sink` port over a torch composite —
    verify-first, the ONNX precedent),
  - `board.py` — `KernelBoard` (the `gamekit.engine.Board` episode;
    `stop` is a scored move),
  - `weights.py` — `KERNEL_PLAYER_WEIGHTS`, the shipped learned prior,
  - `priorart.py` — `KERNEL_PRIOR_ART_RULES` (51 laws), the
    kernel-compilers' known codegen vocabulary as laws (the `pa_kf_*`
    expand bank the novelty sieve derives against),
  - `scan.py` — the `om_chunk_scan` schedule: `build_scan_plan` /
    `is_chunk_scan` term analysis + `ScanRunner`, the pairwise-batched
    forward the `kern_fuse_chunk_scan*` laws lower to,
  - `search.py` — the kernel-IR term search (gate → bound →
    measure; `lib-kernel-ir` `Term`s in, routed `Candidate`s out),
  - `play.py` — the **`kernel` domain plugin**: `register()` seats
    the domain into `kit_discovery.play.DOMAINS` and its players
    into the gamekit registry (importing the module mutates nothing;
    `main()` registers itself) — run
    `python -m mlopt_kernel.play --domain kernel --deliver`.
- `tests/` — `test_kernel.py` (laws, cost, sink, board, play domain)
  plus `test_kernel_priorart.py` / `test_kernel_sieve.py` /
  `test_kernel_search_ir.py` / `test_overnight.py` /
  `test_transfer.py` (175 tests).
- `scripts/train_kernel_prior.py` — retrain + emit the baked prior.
- `bench/` — the measured half: `bench/discovery/kernel.py` (the 46
  kernel cases — `python -m bench.discovery.kernel` from the repo
  root), `bench/limits/triton/` (the hand-Triton field baseline;
  needs `triton` + CUDA, declines honestly without), and
  `bench/drivers/` (the overnight loop, the sieve/prior-art screens,
  the probes, and the transfer driver — result documents, not
  shipped code).
- `docs/` — the measured result documents the claims cite;
  `notes/kernel-engineer-board.md` — the design note + measurement
  ledger; `paper/` — the working draft (proposed, not accepted).

## Develop

The substrate is consumed as **git deps on per-package
subdirectories** of the sibling repos, pinned to the `v0.1.0` tags —
see `[tool.uv.sources]` in `pyproject.toml`:

| dep | repo | source |
|---|---|---|
| `mlopt-core` | `handl-lang/eqsat` | `packages/mlopt-core` |
| `mlopt-orchestrator` | `handl-lang/eqsat` | `packages/mlopt-orchestrator` |
| `kit-torch` | `handl-lang/graphopt` | `packages/kit-torch` |
| `kit-carriers` | `handl-lang/graphopt` | `packages/kit-carriers` |
| `kit-discovery` | `handl-lang/graphopt` | `packages/kit-discovery` |
| `gamekit` | `handl-lang/gamekit` | `packages/gamekit` |
| `benchkit` | `handl-lang/gamekit` | `packages/benchkit` |
| `lib-kernel-ir` | `handl-lang/lib-kernel-ir` | repo root, tag `v0.1.0` |

`mlopt_kernel.search` and its test read `lib_kernel_ir` from the
git dep above; dev worktrees may also resolve a sibling
`../lib-kernel-ir` checkout ahead of the installed wheel (the
`pythonpath`/`extra-paths` entries in `pyproject.toml`), and the
search test skips honestly when `lib_kernel_ir` is absent.  The
rest — the law bank, sink, board, play domain — needs only the git
deps above.

```sh
uv sync                                  # resolves the git deps
```

## Verify

```sh
uv run scripts/gate.py                   # pytest → ruff → ty → lint-imports
.venv/bin/python -m mlopt_kernel.play --domain kernel \
    --train-cases 8 --eval-cases 4 --budget 8   # CLI smoke
```

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` / `LICENSE-APACHE`.
