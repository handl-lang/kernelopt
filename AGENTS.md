# Agent guide — kernelopt

The codegen domain for the mlopt game layer, split out of
`handl-lang/graphopt` (plan 0032 phase 4).  Dual-licensed
MIT OR Apache-2.0.

## Layout

- `src/mlopt_kernel/` — the one shipped package:
  `laws` (the `KERNEL_RULES` codegen bank — fused-kernel ops +
  strength reductions), `cost` (`kernel_cost`), `sink`
  (`KernelSink`/`KernelExecutor` — the `mlopt_core.ports.Sink` port
  over a lazy torch composite, verify-first), `board` (`KernelBoard`
  — the `gamekit.engine.Board` episode), `weights`
  (`KERNEL_PLAYER_WEIGHTS` — the baked prior, retrain via
  `scripts/train_kernel_prior.py`), `priorart`
  (`KERNEL_PRIOR_ART_RULES` — the kernel-compilers' known codegen
  vocabulary as laws; the `pa_kf_*` expand bank the novelty sieve
  derives against), `scan` (the `om_chunk_scan` schedule —
  `build_scan_plan`/`is_chunk_scan` + `ScanRunner`, the
  pairwise-batched forward the `kern_fuse_chunk_scan*` laws lower
  to), `search` (the ADR-0165 kernel-IR term search over
  `lib-kernel-ir` `Term`s — gate → bound → measure), `play` (the
  domain plugin).
- `mlopt_kernel.play` is the seam: `play.register()` seats `kernel`
  into `mlopt_discovery.play.DOMAINS` (`play.spec_domain`) and seats
  `linear`/`linear-greedy`/`linear-cold`/`random`/`playbook` on the
  kernel board (`gamekit.players.registry.register_player` — the
  spec-driven factories need no `make`; `playbook` supplies its own
  because the scripted line lives here).  Registration is explicit —
  importing the module mutates nothing; `main()` calls `register()`
  itself and programmatic consumers call it at their entry point.
- `tests/` — the kernel suite (`test_kernel.py` + the priorart /
  sieve / search-ir / overnight / transfer files).
- `scripts/train_kernel_prior.py` — match-oracle imitation +
  REINFORCE; prints the `weights.py` literal + held-out table.
- `bench/` — measurement, not shipped code: `bench/discovery/kernel.py`
  (46 kernel cases; `python -m bench.discovery.kernel`),
  `bench/limits/triton/` (the hand-Triton baseline; `triton`+CUDA
  only, declines honestly), `bench/limits/registry.py` (re-exports the
  shared `benchkit.baselines` contract types the arm needs).
- `bench/drivers/overnight.py` — the kernel overnight loop; its
  resume ledger is `benchkit.checkpoint.Checkpoint` (the shared
  harness promoted to `handl-lang/gamekit` `packages/benchkit`,
  consumed as the `benchkit` git dep below).
- `docs/`, `notes/`, `paper/` — the measured result documents, the
  design/measurement ledger, and the working paper draft (proposed,
  not accepted) moved out of the parent.

## Substrate deps

Every entry in `[tool.uv.sources]` (`pyproject.toml`) — git deps
pinned to the `v0.1.0` tags on the split sibling repos'
per-package subdirectories, plus one path dep: `mlopt-core` and
`mlopt-orchestrator` (`handl-lang/eqsat`), `mlopt-torch`,
`mlopt-carriers` (the `probes` dep-group only — the vocab probe
lifts the retnet carrier through `mlopt_carriers.om`),
`mlopt-discovery` (`handl-lang/graphopt`), `gamekit` and `benchkit`
(`handl-lang/gamekit` `packages/`), and `lib-kernel-ir` (the
sibling checkout at `../lib-kernel-ir` — `mlopt_kernel.search`
reads its `Term`/`Cost` IR; tests put `../lib-kernel-ir/src` on
`pythonpath` until it is published).  Bump the pin deliberately —
upstream `spec_domain`/`register_player` are the frozen seam this
repo binds; a breaking upstream change shows up as an
import-time failure in `mlopt_kernel.play`.

## Verification

```sh
uv sync
.venv/bin/python -m pytest tests/ -q          # the suite
.venv/bin/ruff check src scripts              # shipped source (parent parity)
.venv/bin/ruff format --check src scripts tests bench
.venv/bin/ty check
.venv/bin/python -m mlopt_kernel.play --domain kernel \
    --train-cases 8 --eval-cases 4 --budget 8  # CLI smoke
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
