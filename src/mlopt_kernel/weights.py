"""Baked player weights — the kernel board's learned prior.

Moved out of ``gamekit.gamedata.weights`` with the kernelopt split
(plan 0032 phase 4): the table is domain data, so it lives where the
domain lives.  Retrain with ``scripts/train_kernel_prior.py`` — it
prints the literal this file pastes.
"""

from __future__ import annotations

#: The kernel board's player prior — v3, trained by
#: ``scripts/train_kernel_prior.py`` on the 23-law bank: imitating
#: the match-then-stop oracle (``state.matches`` rules, then
#: ``stop``) + REINFORCE over 64 kernel cases x 8 epochs; eval (32
#: held-out): trained-greedy 0.630 vs shipped-greedy 0.645 —
#: parity inside the documented band, not a gain; baked for
#: provenance so every rule in the current bank has a trained
#: ``a:r:*`` bucket (v2 covered the 16-law bank; the masked/gqa
#: sdpa variants rode ``x:m`` alone under it).  The v1 table
#: (trained on the 5-law bank, held-out 0.511 > scripted 0.333)
#: carried the first expansion through ``x:m`` alone.  Cold
#: REINFORCE alone still plateaus (cold -0.150 vs scripted 0.226
#: this run) — the imitation phase remains what carries the prior.
KERNEL_PLAYER_WEIGHTS: dict[str, float] = {
    "a:r:00": -0.294,
    "a:r:01": -0.2345,
    "a:r:02": 0.3977,
    "a:r:03": 1.948,
    "a:r:04": -0.6032,
    "a:r:05": 0.4918,
    "a:r:06": -0.4394,
    "a:r:07": 1.3592,
    "a:r:09": -0.3034,
    "a:r:10": -0.7441,
    "a:r:11": -0.384,
    "a:r:12": -0.3747,
    "a:r:13": -0.4141,
    "a:r:14": -0.3466,
    "a:r:15": -0.0587,
    "x:m": 8.6963,
    "x:stop": 1.3816,
    "x:stop:imp": 1.4204,
}
