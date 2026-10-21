"""Baked player weights — the kernel board's learned prior.

Moved out of ``gamekit.gamedata.weights`` with the kernelopt split
(plan 0032 phase 4): the table is domain data, so it lives where the
domain lives.  Retrain with ``scripts/train_kernel_prior.py`` — it
prints the literal this file pastes.
"""

from __future__ import annotations

#: The kernel board's player prior — v2, trained by
#: ``scripts/train_kernel_prior.py`` on the 16-law bank: imitating
#: the match-then-stop oracle (``state.matches`` rules, then
#: ``stop``) + REINFORCE over 64 kernel cases x 8 epochs; eval (32
#: held-out): trained-greedy 0.646 = shipped 0.646 > scripted
#: 0.276 > cold-rl 0.287 > random -0.045.  The v1 table (trained
#: on the 5-law bank, held-out 0.511 > scripted 0.333) carried the
#: expansion through ``x:m`` alone; v2 trains the ``a:r:*`` buckets
#: for every rule in the current bank — a parity retrain, not a
#: gain.  Cold REINFORCE alone still plateaus (~0.29 greedy at
#: 512 episodes) — the imitation phase remains what carries the
#: prior.
KERNEL_PLAYER_WEIGHTS: dict[str, float] = {
    "a:r:01": -0.1974,
    "a:r:02": 0.8806,
    "a:r:03": 1.2169,
    "a:r:05": 0.2912,
    "a:r:06": -0.5514,
    "a:r:07": 0.4942,
    "a:r:10": -0.8337,
    "a:r:11": 0.6186,
    "a:r:12": -0.1206,
    "a:r:13": -0.3816,
    "a:r:14": -0.985,
    "a:r:15": -0.4317,
    "x:m": 7.9501,
    "x:stop": 0.8806,
    "x:stop:imp": 1.62,
}
