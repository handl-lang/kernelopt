# mlopt IR → kernel-IR bridge — Phase C spike

**Question:** can a term `Optimizer.lower` produces (`mlopt_core.ir.Op`
DAG) be translated into a `lib_kernel_ir.Term` for kernel
search/codegen?

**Answer: bridge-viable, but only as a narrow recognizer — not a
general translator.** The shipped spike is
`src/mlopt_kernel/ir_bridge.py` (`translate`), tested by
`tests/test_ir_bridge.py` (28 tests, all green). The honest subset is
**two ops**; everything else declines loudly by name.

## The vocabularies barely intersect

`lib_kernel_ir` is a **closed ten-op** vocabulary (ADR-0165/0166 —
extension requires an ADR), oriented at one kernel shape: the
dequantised GEMV plus its split-k schedule:

| kernel-ir op | role |
|---|---|
| `tile_load` | leaf → tile (identity denote; leaves are costed as this) |
| `unpack_i4`, `dequant` | AWQ weight decode (int4 nibbles → fp tile) |
| `dot` | `einsum("kn,k->n")` — GEMV, contracted dim **first** |
| `split`, `stage`, `reduce` | split-k partition / partial staging / combine |
| `store`, `bias` | write-out epilogue; `tile + bias` elementwise add |
| `pipeline` | schedule-depth annotation (denotes identity) |

`mlopt_core.ir` is a ~95-op general tensor graph (`matmul`, `linear`,
`sdpa`, `rms_norm`, `softmax`, `conv2d`, `split`, `reshape`, … —
plus binding-only spellings in `opmeta._FOLD_AT_LOWERING`).

## Coverage table

| mlopt op(s) | kernel-ir | verdict |
|---|---|---|
| `Var`, `Param` | `Leaf(name)` | **maps** — `k_leaves=` opt-in marks `kind="k"` for split_k slicing; default `"out"` (unsliced, schedule-blind) |
| `matmul(v[k], M[k,n]) → [n]` | `dot(M, v)` (args swapped) | **maps** — the only matmul spelling whose denotation is a `dot`; shape-checked via `TensorType` (rank + contracted dim when known) |
| `matmul(W[m,k], v[k]) → [m]` | — | **declines** — the *common* GEMV spelling needs `Wᵀ`; kernel-ir has no transpose/permute op. Workaround: bind a pre-transposed `[k,m]` weight under a fresh leaf name |
| `matmul` `[m,k]@[k,n]`, bmm, etc. | — | **declines** — not a `dot` |
| `add(tensor, tensor)` | `bias` | **maps** — `bias` denotes `tile + bias`, an exact broadcast add (grade ROUNDS both sides) |
| `add`/`matmul` with `Const`, or any `attrs` | — | **declines** — scalar operand would denote but crash `Term.cost` (`numel`); attrs have no positional-arg counterpart |
| `linear(x, W[out,in], b)` | — | **declines** — `F.linear` contracts `in` against `Wᵀ`; same transpose wall as weight-first matmul |
| `mul`, `sub`, `div`, `relu`, `exp`, `sum`, `mean`, `softmax`, `sdpa`, `rms_norm`, `conv2d`, `layer_norm`, ~90 others | — | **declines** — no kernel-ir counterpart; `UnsupportedOp(op, reason)` names the op |
| `split` | — | **declines — name collision**: mlopt `split` cuts a tensor dim; kernel-ir `split(k_len, parts)` partitions the k axis. Never confused, never mapped |
| `reshape`, `permute`, `transpose`, `flatten`, `squeeze`, `expand`, … | — | **declines** — kernel-ir has no shape/layout ops at all |
| `Const` (bare) | — | **declines** — a scalar is not a tensor program |

## Semantic gaps (substrate-level, not bridge bugs)

- **Layout/transpose:** kernel-ir `dot` stores the contracted dim
  first (`[k,n] @ [k]`). torch/mlopt store weights `[out,in]` or
  `[m,k]`. Without a transpose op, only the `matmul(v, M)` spelling
  bridges — and real exporters emit whichever aten op the model
  used, so coverage is spelling-dependent.
- **Dtype:** kernel-ir `dot`/`dequant` cast operands to fp32 inside
  the denotation; mlopt preserves dtypes (bf16/fp16/int). A bridged
  term denotes in fp32 regardless of the mlopt program's dtype —
  fine for search/cost, wrong if the artifact must match dtype
  exactly.
- **Memory model:** kernel-ir `Leaf.kind` (`"k"`/`"g"`/`"out"`) and
  `split`/`stage`/`reduce` encode a *specific* split-k GEMV schedule
  with an `out` buffer (`store` copies into `env["out"]`). mlopt terms
  are pure — no output buffer, no residency. The bridge emits the
  computational core only; wrapping in `store`/`out` and choosing
  `k_leaves` is the caller's job.
- **Cost model asymmetry:** `Term.cost` prices bytes/ops/launches
  from shapes via `meta` tensors — it works on bridged terms (tested),
  but the op costs were written for the GEMV vocabulary (`bias` costs
  assume a bias-vector add; a large elementwise `add` would be
  under-priced). Cost is a bound, not a ranker (ADR-0165 r6), so this
  is acceptable — but worth knowing.
- **Closed world:** per ADR-0165 the op set is frozen; "the IR needs
  one more op" is a finding about the platform, not a bridge fix.

## Validation performed

`tests/test_ir_bridge.py` (28 tests):

- `add(matmul(v[8], w[8,16]), b[16])` → `bias(dot(w, v), b)`;
  `denote(env)` matches `v @ w + b` under torch; `cost(env)` returns
  sane `Cost`; `to_json`/`from_json` round-trips.
- `k_leaves={"v","w"}` yields `Leaf(kind="k")`; hand-wrapping the
  result in `Term("split", (8, 2, …))` still denotes the full GEMV —
  the split_k correctness property holds through the bridge.
- Every declined family raises `UnsupportedOp` carrying the op name
  and reason (parametrized over 15 op names + shape/attr/const edge
  cases); nested declines propagate the *inner* op's name.

## Verdict: **needs-substrate-work** before `mlopt export --backend kernel`

The bridge machinery is correct and honest, but the coverage is ~2 ops
of ~95. Two paths, in order of leverage:

1. **Extend `lib-kernel-ir`** (its own ADR — the set is deliberately
   closed): a `transpose`/`permute` op alone unlocks weight-first
   GEMV and `linear`; elementwise `mul`/`sub`/`neg`/casts would cover
   the pointwise tail. Until then the bridge can only see
   `matmul(v, M)` + `add` subterms — a thin slice of real decoder
   graphs.
2. **Recognise at a higher level:** a decoder-step `linear`/`matmul`
   + `rms_norm`/`silu` chain won't bridge op-by-op anyway (no
   norm/activation ops). The productive seam is probably
   *pattern → hand-written kernel-ir template* (the `qgemv` builder
   precedent), not term-to-term translation.

Do **not** productize `mlopt export --backend kernel` on this bridge
as-is; the report's coverage table is the honest spec of what would
have to land in `lib-kernel-ir` first.

## Files

- `src/mlopt_kernel/ir_bridge.py` — `translate(term, *, k_leaves=())`,
  `UnsupportedOp`.
- `tests/test_ir_bridge.py` — 28 tests.
- This report.

## Addendum — ADR-0171 vocabulary extension (proposed, pending ratification)

`lib-kernel-ir` gained eleven ops (`transpose`, `permute`, `mul`,
`sub`, `neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu` —
ADR-0171, `host/project/adrs/0171-*.md`, status **proposed**) and the
bridge was extended to match. Coverage moved from **2 ops to 14 mlopt
op names** (`matmul` two spellings, `add`, `linear`, `transpose`,
`permute`, and nine pointwise):

- `matmul(W[m,k], v[k])` → `dot(transpose(W,0,1), v)` — denotes
  `W @ v` exactly (fp32 inside `dot`, as before).
- `linear(x[in], W[out,in], b?)` → `bias(dot(transpose(W), x), b)` —
  rank-1 inputs only; batched inputs still decline.
- `transpose`/`permute` attrs (`dim0`/`dim1`, `dim`/`dims`/`order`)
  become positional int args; `permute()` with no dims reverses all
  axes when the rank is known, declines otherwise.
- Pointwise ops map name-for-name; `Const` operands and attrs still
  decline (cost-model hazard / no positional counterpart).

Still declined: reductions (`sum`/`mean`/`max`), norms, `softmax`,
`sdpa`, `conv*`, `split`/`reshape`/`flatten` and the other ~80 ops —
the ADR's "Not doing" list. `tests/test_ir_bridge.py` is now 45
tests; the previously-declined spellings (weight-first `matmul`,
`linear`, `transpose`, `permute`, the pointwise set) have positive
denotation tests against torch.
