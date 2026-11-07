"""Cross-domain prior transfer — does a trained prior survive a foreign move-set?

The Q5-remainder measurement: the game layer's trained priors are
*weight tables* — ``{feature_name: float}`` rows a
:class:`~gamekit.players.LinearPolicy` dots against whatever the
*target* domain's featurizer emits.  A prior trained on domain A can
reach domain B's board through exactly two seams, and this driver
measures both:

1. **Policy transplant** — bind the *source* spec wholesale
   (``legal`` + ``featurize``) and ask it to play B's states.  The
   boards are structurally different state types (``KernelState`` vs
   ``OpenState`` vs ``MetaState`` vs ``_SearchState``), so the honest
   expectation is the featurizer reads fields that do not exist.
   The compatibility probe below *runs* the attempt and reports the
   exact failure per pair — featurization is the boundary, measured,
   not asserted.

2. **Weight transplant** — the strongest reading of "does the prior
   transfer": keep the *target* featurizer and enumerator (so every
   move is real), and hand the foreign weight table to
   ``LinearPolicy._dot``.  Only weight keys that collide with emitted
   feature names fire at all — and the ``a:r:*`` collision is a *hash
   bucket* (``meta_player._bucket`` — same sha256, different rule
   vocabularies), so even the live slice is semantically scrambled:
   the mass meta trained for rule X lands on whatever kernel rule
   hashes to the same bucket.

Sources are the shipped tables — ``lawdata.META_ARENA_PLAYER_WEIGHTS``,
``lawdata.OPEN_GAME_PLAYER_WEIGHTS``, ``KERNEL_PLAYER_WEIGHTS`` — plus
a ``search`` prior trained at runtime (REINFORCE only, ~2 s): the one
source whose *conventions* overlap the kernel's (``st:steps``,
``st:improve``, the same 16-bucket ``a:r:*`` hash), hence the
strongest possible transfer candidate.

The metric is the **transfer ratio** ``warm ÷ cold`` of held-out
reward/episode on the target board (greedy decode, paired cases);
the honest-null band is reported as ``delta = warm - cold`` too —
cold baselines sit near zero on some boards, where a ratio is
unstable and a delta is the honest number.

Run::

    .venv/bin/python -m bench.drivers.policy.transfer
    .venv/bin/python -m bench.drivers.policy.transfer \
        --eval-cases 16 --open-cases 8 --json /tmp/transfer.json
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from gamekit import engine
from gamekit.players import DomainSpec, LearnedPlayer
from mlopt_discovery import lawdata, play

import mlopt_kernel.play
from mlopt_kernel.weights import KERNEL_PLAYER_WEIGHTS

mlopt_kernel.play.register()  # seat the kernel domain explicitly

#: Held-out stream seed offset — the ``play``/leaderboard convention.
HELD_OUT_OFFSET = 7919

#: Per-domain episode budgets (the domains' own horizons).
BUDGETS: dict[str, int] = {
    "kernel": 8,
    "meta": 24,
    "open": 64,
    "search": 12,
}

#: The shipped priors — ``search`` ships none; it is trained at
#: runtime (REINFORCE) because its schema conventions are the
#: kernel's nearest neighbour.
BAKED_SOURCES: dict[str, dict[str, float]] = {
    "meta": lawdata.META_ARENA_PLAYER_WEIGHTS,
    "open": lawdata.OPEN_GAME_PLAYER_WEIGHTS,
    "kernel": KERNEL_PLAYER_WEIGHTS,
}

#: Cold baselines below this |reward| make the transfer ratio
#: unstable — the report falls back to the delta.
RATIO_FLOOR = 1e-3


def train_search_prior(
    *, seed: int, cases: int, epochs: int, budget: int
) -> dict[str, float]:
    """REINFORCE a search-board prior from the cold start.

    The search spec ships no baked weights, so this driver trains
    one: the only foreign source whose feature conventions
    (``st:steps``, ``st:improve``, the shared ``a:r:*`` hash) line up
    with the kernel schema at all.  Returns the sparse trained table.
    """
    d = play.DOMAINS["search"]
    player = LearnedPlayer(seed, spec=d.spec, weights={})
    stream = d.cases(seed, cases)
    for _ in range(epochs):
        engine.train_policy(player, stream, d.board_of, budget=budget)
    return {
        k: v for k, v in player.weights_dict().items() if abs(v) > 1e-4
    }


def source_weights(
    name: str, *, seed: int, train_cases: int, epochs: int
) -> dict[str, float]:
    """Return the trained weight table for *name* — baked or trained."""
    if name == "search":
        return train_search_prior(
            seed=seed,
            cases=train_cases,
            epochs=epochs,
            budget=BUDGETS["search"],
        )
    return dict(BAKED_SOURCES[name])


def emitted_vocabulary(
    domain: str, *, seed: int, cases: int
) -> set[str]:
    """Feature names the target featurizer emits at step 0.

    The row vocabulary over the first ``cases`` held-out boards'
    initial states — a lower bound on the live set (later states can
    only add keys), honest and oracle-free.
    """
    d = play.DOMAINS[domain]
    assert d.spec is not None
    emitted: set[str] = set()
    for case in d.cases(seed + HELD_OUT_OFFSET, cases):
        state = d.board_of(case).observe()
        hist = d.spec.new_hist()
        for a in d.spec.legal(state):
            emitted.update(d.spec.featurize(state, a, hist))
    return emitted


def vocab_overlap(
    weights: dict[str, float], emitted: set[str]
) -> dict[str, Any]:
    """The live slice of *weights* on rows that only carry *emitted*.

    ``shared`` are the stored weight keys the target featurizer can
    actually fire; ``mass_frac`` is the fraction of the table's
    ``Σ|w|`` they carry — the transplantable share.
    """
    shared = sorted(set(weights) & emitted)
    mass = sum(abs(v) for v in weights.values())
    live = sum(abs(weights[k]) for k in shared)
    return {
        "src_keys": len(weights),
        "shared": shared,
        "live_mass": round(live, 4),
        "total_mass": round(mass, 4),
        "mass_frac": round(live / mass, 4) if mass else 0.0,
    }


def compat(src: DomainSpec, target: str) -> dict[str, str]:
    """Try the source policy's enumerator + featurizer on target states.

    Reports ``ok``/the exception for ``legal`` then ``featurize`` —
    the featurization-boundary claim, measured rather than asserted.
    """
    d = play.DOMAINS[target]
    state = d.board_of(d.cases(HELD_OUT_OFFSET, 1)[0]).observe()
    out = {"legal": "—", "featurize": "—"}
    try:
        acts = src.legal(state)
    except Exception as e:
        out["legal"] = f"{type(e).__name__}: {e}"
        return out
    out["legal"] = f"ok ({len(acts)} moves)"
    if not acts:
        out["featurize"] = "ok (no moves to featurize)"
        return out
    try:
        src.featurize(state, acts[0], src.new_hist())
        out["featurize"] = "ok"
    except Exception as e:
        out["featurize"] = f"{type(e).__name__}: {e}"
    return out


def transfer_ratio(
    warm: float, cold: float
) -> tuple[float | None, float]:
    """Return ``(ratio, delta)``; ratio ``None`` when cold ≈ 0.

    A near-zero cold baseline makes ``warm/cold`` noise rather than a
    measurement — the delta is the honest number there.
    """
    if abs(cold) < RATIO_FLOOR:
        return None, warm - cold
    return warm / cold, warm - cold


def _random_arm(legal: Any, seed: int) -> Any:
    """Uniform-over-legal factory — the floor every target reports."""

    def make(i: int) -> Any:
        r = random.Random(seed + 10_000 + i)

        def go(state: Any) -> Any | None:
            acts = legal(state)
            return acts[r.randrange(len(acts))] if acts else None

        return go

    return make


def _frozen_arm(spec: DomainSpec, weights: dict, greedy: bool) -> Any:
    """A ``LearnedPlayer`` arm factory bound to *spec* + *weights*."""

    def make(i: int) -> Any:
        p = LearnedPlayer(0, spec=spec, weights=weights)
        return p.frozen(20_000 + i, greedy=greedy)

    return make


def eval_target(
    target: str,
    sources: dict[str, dict[str, float]],
    *,
    seed: int,
    eval_cases: int,
    search_spec_xfer: bool = True,
) -> dict[str, Any]:
    """Play the transplant arms on the target's held-out stream.

    Arms: ``random``, ``cold`` (weights={}), ``cold-greedy``, the
    native prior — the domain's baked table, or the runtime-trained
    one when the domain ships none (``search``) — the scripted
    playbook when it seats one, and ``xfer:<src>-greedy`` per
    foreign source.
    On ``kernel`` a bonus arm binds the kernel weights through the
    *search* spec — the only source featurizer that runs on kernel
    states at all (it cannot enumerate ``stop`` — the move the board
    pays for — the deepest form of the boundary).
    """
    d = play.DOMAINS[target]
    assert d.spec is not None
    budget = BUDGETS[target]
    ev = d.cases(seed + HELD_OUT_OFFSET, eval_cases)
    arms: dict[str, Any] = {
        "random": _random_arm(d.spec.legal, seed),
        "cold": _frozen_arm(d.spec, {}, greedy=False),
        "cold-greedy": _frozen_arm(d.spec, {}, greedy=True),
    }
    native = d.spec.weights or sources.get(target)
    if native:
        arms["native-greedy"] = _frozen_arm(
            d.spec, dict(native), greedy=True
        )
    if d.scripted is not None:
        arms["scripted"] = lambda i: d.scripted()
    for src, w in sorted(sources.items()):
        if src == target:
            continue
        arms[f"xfer:{src}-greedy"] = _frozen_arm(d.spec, w, greedy=True)
    if target == "kernel" and search_spec_xfer:
        sspec = play.DOMAINS["search"].spec
        assert sspec is not None
        arms["kern-w@search-spec"] = _frozen_arm(
            sspec, KERNEL_PLAYER_WEIGHTS, greedy=True
        )
    table = engine.evaluate(arms, ev, d.board_of, budget=budget)
    means = {
        name: round(
            sum(r["reward"] for r in rows) / max(len(rows), 1), 4
        )
        for name, rows in table.items()
    }
    cold = means["cold-greedy"]
    rows_out: dict[str, dict[str, Any]] = {}
    for name, mean in means.items():
        ratio, delta = transfer_ratio(mean, cold)
        rows_out[name] = {"reward": mean, "delta": round(delta, 4)}
        rows_out[name]["ratio"] = (
            None if ratio is None else round(ratio, 3)
        )
    return {
        "budget": budget,
        "eval_cases": eval_cases,
        "means": means,
        "arms": rows_out,
    }


def _print_table(title: str, header: str, rows: list[str]) -> None:
    """Print a section header + preformatted rows."""
    print(f"== {title} ==")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(r)
    print()


def main(argv: list[str] | None = None) -> int:
    """Run the compat probe, the overlap table, and the transfer eval."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--eval-cases",
        type=int,
        default=16,
        help="held-out cases per target (open uses --open-cases)",
    )
    ap.add_argument(
        "--open-cases",
        type=int,
        default=8,
        help="held-out cases for the open target (oracle-priced evals)",
    )
    ap.add_argument(
        "--probe-cases",
        type=int,
        default=4,
        help="cases sampled for the emitted-vocabulary probe",
    )
    ap.add_argument(
        "--search-train-cases",
        type=int,
        default=64,
        help="search-prior training stream size",
    )
    ap.add_argument(
        "--search-epochs",
        type=int,
        default=8,
        help="search-prior REINFORCE epochs",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    src_names = ("meta", "open", "kernel", "search")
    targets = ("kernel", "meta", "open", "search")
    weights = {
        s: source_weights(
            s,
            seed=args.seed,
            train_cases=args.search_train_cases,
            epochs=args.search_epochs,
        )
        for s in src_names
    }

    # -- 1. the featurization boundary, measured -------------------
    rows = []
    for s in src_names:
        spec = play.DOMAINS[s].spec
        assert spec is not None
        for t in targets:
            if s == t:
                continue
            c = compat(spec, t)
            rows.append(
                f"{s:>7s} -> {t:<7s} legal: {c['legal']:<42s} "
                f"featurize: {c['featurize']}"
            )
    _print_table(
        "compatibility — source (legal+featurizer) on target states",
        f"{'pair':<17s} {'enumerator':<53s} featurizer",
        rows,
    )

    # -- 2. the transplantable share of each table -----------------
    emitted = {
        t: emitted_vocabulary(t, seed=args.seed, cases=args.probe_cases)
        for t in targets
    }
    rows = []
    overlaps: dict[str, dict[str, Any]] = {}
    for s in src_names:
        overlaps[s] = {}
        for t in targets:
            if s == t:
                continue
            o = vocab_overlap(weights[s], emitted[t])
            overlaps[s][t] = o
            shared = ",".join(o["shared"]) or "(none)"
            rows.append(
                f"{s:>7s} -> {t:<7s} keys {o['src_keys']:>3d} "
                f"live {len(o['shared']):>3d} "
                f"mass {o['mass_frac']:>6.1%}  {shared}"
            )
    _print_table(
        "vocabulary overlap — stored weight keys live on target rows",
        f"{'pair':<17s} {'src':>4s} {'live':>5s} {'mass':>7s}  shared keys",
        rows,
    )

    # -- 3. the transfer eval --------------------------------------
    out: dict[str, Any] = {
        "compat": {},
        "overlap": overlaps,
        "targets": {},
    }
    for t in targets:
        n = args.open_cases if t == "open" else args.eval_cases
        res = eval_target(t, weights, seed=args.seed, eval_cases=n)
        out["targets"][t] = res
        rows = []
        for name, row in res["arms"].items():
            ratio = (
                "n/a(cold~0)"
                if row["ratio"] is None
                else f"{row['ratio']:>7.3f}"
            )
            rows.append(
                f"{name:<22s} reward {row['reward']:>8.3f} "
                f"delta {row['delta']:>8.3f} ratio {ratio}"
            )
        _print_table(
            f"transfer eval — target {t} "
            f"({n} held-out cases, budget {res['budget']})",
            f"{'arm':<22s} {'reward':>9s} {'delta':>9s} {'ratio':>12s}",
            rows,
        )

    print("== verdict ==")
    print(
        "Priors are featurization-bound: cross-domain transplant "
        "survives only through feature-name collisions, and the one "
        "shared channel (a:r:* hash buckets) is semantically "
        "scrambled — see docs/results_transfer.md."
    )
    if args.json:
        out["weights"] = weights
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
