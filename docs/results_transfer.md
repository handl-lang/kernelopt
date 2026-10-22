# Cross-domain prior transfer — does a trained prior survive a foreign move-set?

Q5 remainder.  The game layer's trained priors are *weight tables* —
`{feature_name: float}` rows a `LinearPolicy` dots against the feature
row the *target* domain's featurizer emits.  A prior trained on
domain A reaches domain B through exactly two seams, and this
measurement runs both:

1. **Policy transplant** — bind the source `DomainSpec` wholesale
   (`legal` + `featurize`) and ask it to play B's states.
2. **Weight transplant** — the strongest reading of "the prior
   transfers": keep the *target* featurizer/enumerator (every move is
   real), hand the foreign weight table to the linear scorer.  Only
   weight keys that *collide by name* with emitted feature rows fire
   at all.

Driver: `bench/drivers/policy/transfer.py` — reproduce with

    .venv/bin/python -m bench.drivers.policy.transfer \
        --json /tmp/transfer.json

Sources: `lawdata.META_ARENA_PLAYER_WEIGHTS` (the meta prior —
shared spec across `meta`/`joint`/`torch`/`gen`),
`lawdata.OPEN_GAME_PLAYER_WEIGHTS`, `KERNEL_PLAYER_WEIGHTS`, plus a
`search` prior trained at runtime (REINFORCE, 64 cases × 8 epochs,
~2 s — the one source whose *conventions* overlap the kernel's:
`st:steps`, `st:improve`, the same 16-bucket `a:r:*` hash, so the
strongest possible transfer candidate).

Protocol: held-out stream `cases(seed + 7919, n)` — the
`play`/leaderboard convention — greedy decode, paired boards, each
domain's own budget (kernel 8, meta 24, open 64, search 12).
`ratio` is `warm ÷ cold-greedy`; `delta` the honest fallback where
cold sits near zero.  Box caveat: load average 5–7 during the run
(shared box) — rewards are deterministic under fixed seeds; only
wall-clock is noisy.

## 1. The featurization boundary — measured, not asserted

Source `legal(state)` then `featurize(state, action, hist)` on one
held-out target state:

| source → target | enumerator | featurizer |
|---|---|---|
| meta → kernel | `AttributeError: 'KernelState' … 'rules'` | — |
| meta → open   | `AttributeError: 'OpenState' … 'rules'`   | — |
| meta → search | `AttributeError: '_SearchState' … 'rules'` | — |
| open → kernel | `AttributeError: 'KernelState' … 'pending'` | — |
| open → meta   | `AttributeError: 'MetaState' … 'pending'`  | — |
| open → search | `AttributeError: '_SearchState' … 'pending'` | — |
| kernel → meta | `AttributeError: 'MetaState' … 'actions'`  | — |
| kernel → open | `AttributeError: 'OpenState' … 'actions'`  | — |
| kernel → search | ok (75 moves) | `AttributeError: '_SearchState' … 'has_const'` |
| search → kernel | ok (26 moves) | **ok** |
| search → meta | `AttributeError: 'MetaState' … 'actions'` | — |
| search → open | `AttributeError: 'OpenState' … 'actions'` | — |

**10 of 12 foreign pairs cannot even enumerate a move.**  The two
that can are the `search`/`kernel` pair — `KernelState` is a field
superset of `_SearchState` — and only in one direction: the search
spec *runs* on kernel states, but its enumerator offers only
`state.actions` — the kernel board's `stop`, the move the board pays
for, is invisible to it.  The source policy as a *policy* does not
transfer, period.

## 2. The transplantable share of each table

Stored weight keys that fire on the target's step-0 feature rows, as
a share of the table's `Σ|w|`:

| source → target | stored keys | live keys | live mass |
|---|---|---|---|
| meta → kernel  | 42 | 15 | 17.1% — `a:r:*` only |
| meta → open    | 42 | 0 | 0.0% |
| meta → search  | 42 | 15 | 18.7% — `a:r:*` only |
| open → kernel  | 37 | 0 | 0.0% |
| open → meta    | 37 | 0 | 0.0% |
| open → search  | 37 | 0 | 0.0% |
| kernel → meta  | 15 | 11 | 35.1% — `a:r:*` only |
| kernel → open  | 15 | 0 | 0.0% |
| kernel → search| 15 | 11 | 35.1% — `a:r:*` only |
| search → kernel| 15 | 14 | 96.8% — `a:r:*` only |
| search → meta  | 15 | 15 | 100.0% — `a:r:*` only |
| search → open  | 15 | 0 | 0.0% |

The `open` vocabulary shares **zero** feature names with every other
domain — its schema is candidate-shape (`a:form:*`, `a:score*`) and
verdict counters; `bias`/`st:steps` are untrained zeros in its baked
table.  The `meta`/`kernel`/`search` overlap is *one channel*:
`a:r:NN` — the rule-name hash bucket (`meta_player._bucket`, sha256
mod 16).  Same hash, **different rule vocabularies**: the mass meta
trained for a tensor-bank rule lands on whatever kernel rule hashes
into the same bucket.  It is a name collision, not a shared
semantics — the live slice is semantically scrambled by
construction.

## 3. Held-out reward/episode — the transfer ratio

Target **kernel** (16 cases, budget 8):

| arm | reward | Δ vs cold | ratio |
|---|---|---|---|
| random | −0.101 | +0.028 | 0.780 |
| cold (sampled) | −0.092 | +0.037 | 0.715 |
| cold-greedy | −0.129 | 0 | 1.000 |
| **native-greedy** (kernel prior) | **+0.653** | +0.782 | — |
| scripted (playbook) | +0.289 | +0.418 | — |
| xfer:meta-greedy | −0.050 | +0.079 | 0.387 |
| xfer:open-greedy | −0.129 | 0.000 | 1.000 |
| xfer:search-greedy | −0.272 | **−0.143** | — |
| kern-w@search-spec | −0.035 | +0.094 | 0.275 |

Target **meta** (16 cases, budget 24):

| arm | reward | Δ vs cold | ratio |
|---|---|---|---|
| random | −0.859 | +0.341 | 0.716 |
| cold (sampled) | −0.738 | +0.462 | 0.615 |
| cold-greedy | −1.200 | 0 | 1.000 |
| **native-greedy** (meta prior) | **+0.117** | +1.317 | — |
| scripted | −0.100 | +1.100 | — |
| xfer:kernel-greedy | −1.200 | 0.000 | 1.000 |
| xfer:open-greedy | −1.200 | 0.000 | 1.000 |
| xfer:search-greedy | −1.200 | 0.000 | 1.000 |

Target **open** (8 cases, budget 64 — oracle-priced evals):

| arm | reward | Δ vs cold | ratio |
|---|---|---|---|
| random | +0.859 | −0.732 | 0.540 |
| cold (sampled) | +0.520 | −1.071 | 0.327 |
| cold-greedy | +1.591 | 0 | 1.000 |
| **native-greedy** (open prior) | **+1.868** | +0.277 | — |
| xfer:kernel-greedy | +1.591 | 0.000 | 1.000 |
| xfer:meta-greedy | +1.591 | 0.000 | 1.000 |
| xfer:search-greedy | +1.591 | 0.000 | 1.000 |

Target **search** (16 cases, budget 12):

| arm | reward | Δ vs cold | ratio |
|---|---|---|---|
| random | +0.125 | +0.125 | n/a (cold≈0) |
| cold (sampled) | +0.062 | +0.062 | n/a |
| cold-greedy | 0.000 | 0 | n/a |
| **native-greedy** (trained at runtime) | **+0.312** | +0.312 | n/a |
| scripted | +0.667 | +0.667 | n/a |
| xfer:kernel-greedy | 0.000 | 0.000 | n/a |
| xfer:meta-greedy | 0.000 | 0.000 | n/a |
| xfer:open-greedy | 0.000 | 0.000 | n/a |

## 4. Reading

**No foreign prior transfers.**  Every `xfer` arm lands at or below
its cold baseline:

- *Zero-overlap directions are literally cold*: `open`→anything and
  anything→`open` reproduce the cold-greedy reward **bit-for-bit**
  — with zero live keys the transplant is `weights={}`, confirmed by
  the reward, not just the overlap math.
- *The `a:r:*`-overlap directions are cold or worse*: on `meta` and
  `search` the transplants are bit-identical to cold (the bucketed
  weights never move the argmax off the cold trajectory); on
  `kernel`, the search-trained table — the source whose conventions
  most resemble the target's — is **actively harmful** (−0.143):
  the hash-scrambled rule ordering fires losing laws first.  The
  meta prior's +0.079 bump on kernel is the same noise channel in a
  luckier direction, 0.70 below the native prior's reward.
- *Even the working cross-application fails the board's semantics*:
  `kern-w@search-spec` runs — the kernel table's `a:r:*` ordering
  fires matching rules early (+0.094 over cold) — but the search
  enumerator has no `stop` to offer, so the arm pays step-cost to
  the horizon and nets −0.035 where the native prior banks +0.653.

The anchors confirm the protocol is sound: every native prior clears
its cold baseline on its own board (kernel +0.782, meta +1.317,
open +0.277, search +0.312 reward delta).

## 5. Verdict — and its one honest sharpening

**Priors are featurization-bound; the transfer ratio is undefined —
or degenerate — without a shared feature contract.**  A weight table
is not a policy: the policy is `(legal, featurize, weights)`, and
both bound parts are per-domain.  What can cross a repo boundary is
only the name-keyed mass, and the game layer's vocabularies were
grown per-domain (`st:steps` is `steps/8` on kernel and search but
`steps/32` on meta; `x:m`, `x:stop:imp`, the whole `a:form:*`/
`x:score:*` block exist in exactly one schema).

The sharpening the architecture itself supplies: a prior is bound
to a **spec**, not a domain name.  `META_SPEC` genuinely plays four
boards (`meta`/`joint`/`torch`/`gen`) — that *is* transfer, and it
works precisely because those domains share the move vocabulary,
the featurizer, and the state type the spec was built on.  The
boundary the measurement finds is the spec's: same spec ⇒ same
prior plays; different spec ⇒ featurization-bound, transfer
undefined.

### Caveats

- Reward scales differ per board; `ratio` is reported beside
  `delta` and suppressed where cold ≈ 0 (search).  Sample sizes are
  the driver's defaults (16/8 cases) — enough for bit-identical
  zero-transfer rows and a −0.143 harm signal, not for fine
  distinctions inside the noise band around cold.
- The `a:r:*` channel is the *only* shared vocabulary, and it is a
  hash: a same-name weight on a different rule is noise, not
  semantics.  A real shared-schema experiment (e.g. training one
  spec over two move-sets deliberately) is a different design —
  this measurement asks what the shipped machinery transfers, and
  the shipped answer is nothing.
- The `search` source prior is trained at runtime (the domain ships
  none); its native anchor (+0.312 vs cold 0.000) confirms the
  source is a real prior, not a null table.
