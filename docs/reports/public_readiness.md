# Public-readiness audit — kernelopt

Date: 2026-11-13.  Scope: the tracked tree only (59 files at audit
time), current checkout — no history rewrite performed or required.

## Verdict: **CONDITIONAL GO**

The code, packaging, docs, and license are public-clean and all gates
pass.  Two decisions belong to the owner before flipping visibility —
see "Remaining risks" #1 and #2.  Neither is a code defect; both are
content/scope calls.

## What was fixed

- **License added.**  No LICENSE existed.  Copied the dual-license
  pair verbatim from `compile-triton` (`LICENSE-MIT` +
  `LICENSE-APACHE`, "the handl-lang contributors") and set
  `license = "MIT OR Apache-2.0"` (PEP 639 SPDX expression) in
  `pyproject.toml`; bumped `setuptools>=68` → `>=77` for license-
  expression support.  Note: siblings are mixed — `graphopt` and
  `host-tensor` ship MIT-only `LICENSE`; `compile-triton` is the dual
  precedent this follows.
- **README rewritten for a public reader.**  Removed the "**private**"
  banner and the "nothing may be referenced" warning; added what the
  repo *is* (the codegen domain split from graphopt, plan 0032 phase
  4), the license line, verified counts (28 `KERNEL_RULES`, 51
  `KERNEL_PRIOR_ART_RULES`, 46 bench cases, 175 tests — all measured
  against source), and a `docs/`+`notes/`+`paper/` pointer.  Fixed a
  real stale claim: it said *importing* `play.py` seats the domain —
  registration is explicit (`register()`; `main()` calls it).
  Documented the `lib-kernel-ir` sibling requirement.
- **`pyproject.toml` description** de-"privated"; **`AGENTS.md`**
  header/deps sections updated (public repo, `v0.1.0` tag pins, the
  `paper/` dir acknowledged).
- **`tests/test_kernel_search_ir.py`** now `pytest.importorskip`s
  `lib_kernel_ir` — a public clone without the private sibling skips
  honestly instead of erroring at collection.
- **`pyproject.toml` `[tool.ty.environment].extra-paths`** gained
  `../lib-kernel-ir/src` (mirrors the pytest `pythonpath` entry) —
  `ty check` was failing with `unresolved-import: lib_kernel_ir`
  before this; now passes.
- **`src/mlopt_kernel/board.py`** docstring: dropped the
  "private-repo extraction" rationale.
- **`ci.sh`** comment: "Private git deps" → accurate fallback wording
  (deps are now public https git URLs).
- **`docs/results_kernel_check.md`**: fixed a dead pointer
  (`project/notes/…` → `notes/…`; `project/` is gitignored).
- **`docs/results_prior_art_kernel.md`**: added a snapshot note — the
  doc reports 49 laws; the bank is 51 today (the two `sdpa_mask`/`gqa`
  denotations landed after the run).
- **`uv.lock` regenerated** (`uv lock`) after the `branch = "main"` →
  `tag = "v0.1.0"` re-pin; it still recorded the old branch URLs.
  `uv sync` verified against the public https tags — no ssh or
  credentials needed.

## Gates run (all green)

- `.venv/bin/python -m pytest tests/ -q` — **175 passed** (~4 min,
  on the re-pinned v0.1.0 deps).
- `.venv/bin/ruff check src scripts` — clean; `ruff format --check
  src scripts tests bench` — clean.
- `.venv/bin/ty check` — clean (0 errors).

## Secrets / sensitive scan — clean

- No `.env*`, keys, tokens, credentials, or private URLs in tracked
  files; the only URL families are github.com/handl-lang (public
  repos), pypi/files.pythonhosted.org, and one fla-org reference.
- `bupha.toml` contains `dir = "/home/luis/GitHub/handl-lang/
  kernelopt"` — an absolute path leaking the username.  **Left as-is
  deliberately**: `compile-triton/bupha.toml` and
  `lib-kernel-ir/bupha.toml` carry the identical convention, and the
  file is the committed manifest for the internal `bupha` CI watcher
  (sanitizing the path would break the runner).  Low risk — `luis`
  is already public via graphopt's LICENSE copyright.
- `myco.toml` — internal task-ledger config, no secrets; `gamekit`
  and `graphopt` ship the same file.
- No `.github/workflows`; CI is `ci.sh` + `bupha.toml`, both of which
  work from a plain public clone (`uv sync` needs only https).

## Remaining risks / decisions for the owner

1. **`paper/` and `notes/` are marked PRIVATE and say the numbers are
   "repo-internal — do not copy into public trees"; the note says the
   repo "stays private until it earns publication."**  Flipping
   visibility publishes the working paper draft (`paper.pdf`,
   `paper.qmd`) and the internal measurement ledger as-is.  If that is
   intended, the banners are now stale-but-harmless; if not, strip or
   unlist `paper/` + `notes/` before flipping.  **This is the only
   real GO/NO-GO question.**
2. **`handl-lang/lib-kernel-ir` is still a PRIVATE repo.**  It is a
   path dep (`../lib-kernel-ir`) for `mlopt_kernel.search` + its
   test.  Consequences for a public clone: `uv sync` and the whole
   suite pass (search tests skip), but `mlopt_kernel.search` is
   unusable and `ty check` errors on the unresolved import unless the
   sibling is checked out next door.  Cleanest fix: make
   lib-kernel-ir public too and switch the source to a git+tag dep;
   alternatively accept the degraded surface and say so in the
   README (already noted there).
3. **Concurrent commits swept up in-flight edits.**  Two commits
   landed mid-audit from a parallel session: `be71e28` ("build: pin
   git deps to v0.1.0 tags") committed this audit's `README.md` +
   `pyproject.toml` license/description edits under its own message,
   and `72ee945` ("chore: relock against v0.1.0 tags") committed the
   regenerated `uv.lock`.  Content is correct in both cases; the
   history is just misleading.  Remaining audit edits (AGENTS.md,
   board.py, ci.sh, the search-test skip, docs notes, LICENSE files,
   this report) are still uncommitted — per task rules nothing was
   committed by this pass.
4. **Historical docs cite parent-repo commits/paths**
   (`packages/mlopt-kernel/`, commit `3801504`, graphopt-side layout)
   — accurate as citations of the pre-split tree; left unchanged.
5. `docs/results_*.md` are dated measurement reports; counts inside
   them are snapshots (two got explicit snapshot notes where the
   drift would mislead).

## Recommendation

**GO once the owner answers two questions:** (a) are `paper/` +
`notes/` meant to be public (or should they be stripped first)? and
(b) is `lib-kernel-ir` going public / should kernelopt ship without
`mlopt_kernel.search` until it does?  Everything else — license,
README, deps, CI, gates — is ready.
