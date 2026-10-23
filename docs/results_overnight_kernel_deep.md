# Kernel overnight discovery run

Episodes -> mint -> admit -> classify on the kernel corpus.  Each episode's arena pool is minted *from the case's own term*: the shared corpus-census proposals that bind it plus the ``syn_*`` pair-composites ``meta.synthesize_rules`` mints over ``KERNEL_RULES`` (+ the substrate's guarded parents) seeded on the term.  Every mint faces the emit gate (strict LHS bind + instantiated RHS must denote).  Evaluation binds a firing site, verifies the concrete pair through ``KernelSink`` (eager bodies, fp64) and prices the firing in ``kernel_cost``.  The admission gauntlet is the kernel referee — ``oracle`` (instance evaluates ``equal``), ``pay`` (the rule alone must extract a *verified* term strictly below the spelled cost on a registered case), ``sweep`` (all bound corpus sites equal), ``cert`` (the instance carries a standalone-replayable certificate).  Admitted rules are classified by the novelty sieve's gates — **review** (cert ∧ beat ∧ ¬derivable: the bank cannot prove it) is the machine-original class.

- date: 2026-10-23 02:26 UTC · git: `dd80062` · torch 2.14.1+cu130 on cuda
- protocol: `{"budget": 24, "cases": null, "dry": false, "episodes": 10, "fuel": 256, "mint": "kernel-corpus-compose", "page": 24, "player": "playbook", "referee": "full", "rounds": 4, "seed": 0, "vocab": "hand"}`
- episodes: 10 · evaluated: 3 · minted: 3 (1 composites, 1 guarded) · admitted: 2 · wall: 7.6s
- arena verdicts: known=2, useful=1

## Admission funnel

| first-failed leg | count |
|---|---|
| pay | 1 |
| **admitted** | 2 |

## Novelty

| verdict | count |
|---|---|
| known | 1 |
| unpaid | 1 |

## Machine-original finds (sieve `review`)

**None.** Every admitted mint was already a theorem of the known bank — or nothing survived the gauntlet.  The funnel above is the measurement: composed kernel rules are derivable by construction when both parents are in the bank, so the honest expectation is refusals at ``pay`` (a composite never beats the floor its own chain reaches) and ``known`` verdicts at classify.

## Candidates

| candidate | family | arena | admitted | reason | novelty |
|---|---|---|---|---|---|
| `recognize:softmax` | pattern-recognition | known | yes | cleared | unpaid |
| `grammar:sub_to_add_dup` | algebraic-grammar | known | no | pay: no verified cost-improving extraction | — |
| `syn_naturality_scalar_rev__kern_fuse_mm_bias_0@kernel_mm_scale_bias` | synth-composite | useful | yes | cleared | known |
