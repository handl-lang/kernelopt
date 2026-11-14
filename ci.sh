#!/usr/bin/env bash
# ci.sh — kernelopt CI gate for the bupha shell runner.
#
# uv-managed Python project: sync the locked env, run the pytest suite.
# Deps are public git URLs ([tool.uv.sources]); the worktree plumbing
# below recreates the sibling layout because pytest pythonpath /
# ty extra-paths resolve ``../lib-kernel-ir/src`` for dev checkouts.
set -euo pipefail

cd "$(dirname "$0")"

# Workspace root = parent of the *main* checkout (correct from both the
# checkout and any detached worktree of this repo).
GIT_COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"
MAIN_CHECKOUT="$(dirname "$GIT_COMMON")"
WORKSPACE_ROOT="$(dirname "$MAIN_CHECKOUT")"

if [ "$PWD" != "$MAIN_CHECKOUT" ]; then
  parent="$(cd .. && pwd)"
  if [ "$parent" = "$WORKSPACE_ROOT" ]; then
    echo "ci.sh: CI worktree sits inside $WORKSPACE_ROOT — ../kernelopt" >&2
    echo "       would resolve to the dev checkout instead of this sha." >&2
    echo "       Move the worktree base outside the workspace root." >&2
    exit 1
  fi
  if [ -e "$parent/kernelopt" ] && [ ! -L "$parent/kernelopt" ]; then
    echo "ci.sh: $parent/kernelopt is a real directory — refusing to clobber" >&2
    exit 1
  fi
  ln -sfn "$PWD" "$parent/kernelopt"
  for d in "$WORKSPACE_ROOT"/*/; do
    name="$(basename "$d")"
    [ "$name" = "kernelopt" ] && continue
    if [ -L "$parent/$name" ] || [ ! -e "$parent/$name" ]; then
      ln -sfn "$d" "$parent/$name"
    fi
  done
fi

# Fallback for ssh-style remotes: insteadOf rewrite scoped to this
# process tree — never touches ~/.gitconfig.
if [ -z "${GIT_CONFIG_COUNT:-}" ]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.git@github.com:handl-lang/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/handl-lang/"
fi

uv sync
uv run pytest tests/ -q
