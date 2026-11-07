# Agent guide — kernelopt (PRIVATE)

The codegen domain for the mlopt game layer — a **private** repo,
split out of `handl-lang/graphopt` (plan 0032 phase 4).
Nothing in this repo may leak into the parent's public docs/paper.

## Layout

- `src/mlopt_kernel/` — the one shipped package:
  `laws` (the `KERNEL_RULES` codegen bank — fused-kernel ops +
  strength reductions), `cost` (`kernel_cost`), `sink`
  (`KernelSink`/`KernelExecutor` — the `mlopt_core.ports.Sink` port
  over a lazy torch composite, verify-first), `board` (`KernelBoard`
  — the `gamekit.engine.Board` episode), `weights`
  (`KERNEL_PLAYER_WEIGHTS` — the baked prior, retrain via
  `scripts/train_kernel_prior.py`), `play` (the domain plugin).
- `mlopt_kernel.play` is the seam: `play.register()` seats `kernel`
  into `mlopt_discovery.play.DOMAINS` (`play.spec_domain`) and seats
  `linear`/`linear-greedy`/`linear-cold`/`random`/`playbook` on the
  kernel board (`gamekit.players.registry.register_player` — the
  spec-driven factories need no `make`; `playbook` supplies its own
  because the scripted line lives here).  Registration is explicit —
  importing the module mutates nothing; `main()` calls `register()`
  itself and programmatic consumers call it at their entry point.
- `tests/` — the kernel suite (`test_kernel.py`).
- `scripts/train_kernel_prior.py` — match-oracle imitation +
  REINFORCE; prints the `weights.py` literal + held-out table.
- `bench/` — measurement, not shipped code: `bench/discovery/kernel.py`
  (44 kernel cases; `python -m bench.discovery.kernel`),
  `bench/limits/triton/` (the hand-Triton baseline; `triton`+CUDA
  only, declines honestly), `bench/limits/registry.py` (re-exports the
  shared `benchkit.baselines` contract types the arm needs).
- `bench/drivers/overnight.py` — the kernel overnight loop; its
  resume ledger is `benchkit.checkpoint.Checkpoint` (the shared
  harness promoted to `handl-lang/gamekit` `packages/benchkit`,
  consumed as the `benchkit` git dep below).
- `docs/`, `notes/` — the audit + design note moved out of the
  parent (they are why this repo is private).

## Substrate deps

uv git+ssh deps on `handl-lang/graphopt` subdirectories, pinned
to `main` (`[tool.uv.sources]`): `mlopt-core`, `mlopt-torch`,
`mlopt-orchestrator`, `gamekit`, `mlopt-discovery`.  Bump the pin
deliberately — upstream `spec_domain`/`register_player` are the
frozen seam this repo binds; a breaking upstream change shows up as
an import-time failure in `mlopt_kernel.play`.

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
