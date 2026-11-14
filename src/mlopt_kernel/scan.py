"""``mlopt_kernel.scan`` — the chunked decayed-scan schedule.

The ``om_chunk_scan`` op's two halves, vendored carrier-free (the
kernel package's standing posture — ``_decay``/``_decay_vec``/
``_om_decayed`` in the sink are the same port): the term analysis
and the pairwise-batched forward of the parent's
``ChunkedDecayModule`` (``mlopt_carriers.decayed_lower``), restated
over the kernel executor's ``ev(t)`` operand channel instead of an
embedded ``IRModule``.

The pattern the laws mint and this module schedules — a
``concat(dim)`` over per-chunk decayed sums::

    out_i = om_decayed(s_i, decay, v_i) / S
          + Σ_j  row_ij ⊙ (q_i @ (A_j @ v_j)) / S

—the fully expanded pairwise-block form of the chunked decayed
scan: chunk *i*'s output gathers a masked intra-chunk attention
element plus one decayed cross term per earlier block *j*.  The
spelled tape launches each piece per chunk; the scheduled form
batches every operand across the chunk axis:

* **Intra** — the score matmuls batch into one ``matmul`` over the
  stacked chunk axis, and so do the decay product and the ``@v``
  contraction: 2 batched GEMMs for what was 2K serial ones.
* **Cross** — the ``A_j @ v_j`` state halves are content-identical
  across target chunks (interned terms dedup) and batch into one
  GEMM; every pair product ``q_i @ M_j`` gathers into a second.
  Contributions accumulate per target chunk by deterministic
  segment sums (pairs arrive in arm order — contiguous per chunk),
  never atomic scatter.
* **Slices** — the per-chunk operands are ``split``/``select``
  views of shared projections; :func:`_sliced_gather` restacks them
  as reshape+permute views, zero kernels where the spelled tape
  copied.
* **Constants** — the ``decay``/``decay_vec`` gates carry only
  Param/Const leaves; the executor's static-subtree memo already
  evaluates them once per lowering (the carrier's ``_const``
  schedule choice, inherited).

:func:`build_scan_plan` detects the pattern — an ``om_chunk_scan``
(or the ``concat`` site it folds) whose flattened arms each parse
as one intra element plus decay-gated pair products, under uniform
concrete shapes.  Anything else returns ``None`` and the executor
falls back to the spelled per-arm evaluation — a drop-in, never a
silent drop.  :func:`is_chunk_scan` is the law check's half.

The intra classifier accepts both spellings of the decayed element
— the fused ``om_decayed(s, d, v)`` enode *and* the raw
``matmul(mul(s, d), v)`` site ``kern_fuse_om_decayed`` mints it
from — so a chunked scan written without the fold still schedules.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ir import Const, Op
from mlopt_core.typing import _shape_of as _shape_of_raw

__all__ = [
    "ScanRunner",
    "build_scan_plan",
    "is_chunk_scan",
    "scan_arms",
]


def _torch() -> Any:
    """Return torch lazily — same layering posture as the sink."""
    import torch

    return torch


#: Content-keyed shape memo — bound terms are interned ``Op``
#: objects; threading one memo turns repeated shape walks into
#: lookups (the ``laws.base._SHAPE_MEMO`` convention).
_SHAPES: dict = {}


def _shape_of(t: Any) -> Any:
    """Memoised shape lookup over the scan parser's term sets."""
    return _shape_of_raw(t, _SHAPES)


# ---------------------------------------------------------------------------
#  Term plumbing — the arm parser
# ---------------------------------------------------------------------------


def _flat_same(t: Any, op: str, dim: int | None = None) -> list:
    """Flatten a same-op tree (``concat`` on *dim*, or ``add``)."""
    if (
        isinstance(t, Op)
        and t.op == op
        and len(t.args) >= 2
        and (dim is None or t.attrs.get("dim") == dim)
    ):
        out: list = []
        for a in t.args:
            out += _flat_same(a, op, dim)
        return out
    return [t]


def scan_arms(t: Any, dim: int) -> list:
    """Flatten *t* into its chunk arms along *dim*.

    Descends through ``concat`` (the spelled site) and
    ``om_chunk_scan`` (the minted marker — a nested fold) nodes whose
    ``dim`` attr matches; everything else is a leaf arm.  The arm
    order is concat order — the chunk sequence.
    """
    if (
        isinstance(t, Op)
        and t.op in ("concat", "om_chunk_scan")
        and len(t.args) >= 2
        and t.attrs.get("dim") == dim
    ):
        out: list = []
        for a in t.args:
            out += scan_arms(a, dim)
        return out
    return [t]


def _decay_row(t: Any) -> Op | None:
    """Match ``unsqueeze(decay_vec(...), dim)`` — a per-row decay gate."""
    if (
        isinstance(t, Op)
        and t.op == "unsqueeze"
        and len(t.args) == 1
        and isinstance(t.args[0], Op)
        and t.args[0].op == "decay_vec"
    ):
        return t
    return None


def _gated_factor(a: Any, b: Any) -> tuple | None:
    """Return the ``(gate, other)`` of a two-operand product, or ``None``.

    Exactly one operand must be a :func:`_decay_row`; zero or two
    gates is not a gated product.
    """
    u0, u1 = _decay_row(a), _decay_row(b)
    if (u0 is None) == (u1 is None):
        return None
    return (u0, b) if u0 is not None else (u1, a)


def _peel_summand(t: Any) -> tuple | None:
    """Peel ``mul(decay_row, ·)``/``div(·, const)`` off a summand.

    Returns ``(row_term | None, body, scale_leaf | None)`` — the row
    gate may sit inside or outside the ``div``, and either ``mul``
    operand order is accepted.  ``None`` when peeling lands on a
    non-``Op`` body or a second row gate appears.
    """
    row: Op | None = None
    scale: Any = None
    while isinstance(t, Op):  # every pass shrinks t or breaks
        if (
            t.op == "div"
            and len(t.args) == 2
            and not isinstance(t.args[1], Op)
            and scale is None
        ):
            scale = t.args[1]
            t = t.args[0]
            continue
        if t.op == "mul" and len(t.args) == 2:
            gate = _gated_factor(t.args[0], t.args[1])
            if gate is None:
                break  # no row gate — or two: the mul IS the body
            if row is not None:
                return None
            row, t = gate
            continue
        break
    if not isinstance(t, Op):
        return None
    return (row, t, scale)


def _classify_body(t: Op) -> tuple | None:
    """Classify a peeled summand body.

    ``("intra", (score, decay, v))`` for ``om_decayed(s, d, v)`` — or
    its spelled site ``matmul(mul(s, d), v)`` — and
    ``("cross", (q, a@v))`` for the right-associated pair product
    ``q @ (A @ v)``, the extracted decayed-scan's cross term.  The
    ``matmul``-rhs check runs first: ``matmul(mul(s,d), A@v)`` is a
    cross term whose q happens to be a product, never an intra.
    """
    if t.op == "om_decayed" and len(t.args) == 3:
        return ("intra", tuple(t.args))
    if t.op == "matmul" and len(t.args) == 2:
        inner = t.args[1]
        if (
            isinstance(inner, Op)
            and inner.op == "matmul"
            and len(inner.args) == 2
        ):
            return ("cross", (t.args[0], inner))
        lhs = t.args[0]
        if (
            isinstance(lhs, Op)
            and lhs.op == "mul"
            and len(lhs.args) == 2
        ):
            return ("intra", (lhs.args[0], lhs.args[1], t.args[1]))
    return None


def _parse_arm(arm: Any) -> dict | None:
    """Parse one chunk arm into ``{intra, cross[]}`` or ``None``.

    Every arm must carry exactly one decayed intra element; every
    other summand must be a decay-gated pair product.
    """
    intra: tuple | None = None
    cross: list[tuple] = []
    for s in _flat_same(arm, "add"):
        p = _peel_summand(s)
        if p is None:
            return None
        row, body, scale = p
        cls = _classify_body(body)
        if cls is None or (cls[0] == "intra" and intra is not None):
            return None
        if cls[0] == "intra":
            intra = (*cls[1], scale, row)
        else:
            cross.append((row, *cls[1], scale))
    if intra is None:
        return None
    return {"intra": intra, "cross": cross}


def _uniform(shapes: list) -> tuple | None:
    """Return the common concrete shape, or ``None`` when non-uniform."""
    s0 = shapes[0]
    if not (
        isinstance(s0, tuple)
        and all(isinstance(d, int) for d in s0)
        and all(s == s0 for s in shapes[1:])
    ):
        return None
    return s0


def _scalar_leaf(t: Any) -> bool:
    """Return True when a scale leaf can become a float at run time.

    ``ScanRunner._scalar`` accepts ``None`` (the peeled ``div``-less
    summand — an exact 1.0), int/float literals, ``Const`` leaves and
    rank-0 (``()``-shaped) terms; anything else raises *inside*
    ``run`` after the plan has committed and the spelled fallback is
    gone.  Rejecting here keeps ``build_scan_plan`` honest: a plan
    whose divisors aren't provably scalar never forms.
    """
    if t is None or isinstance(t, (int, float)):
        return True
    if isinstance(t, Const):
        return isinstance(t.value, (int, float))
    return _shape_of(t) == ()


def _scales_safe(arms: list[dict]) -> bool:
    """Return True when every intra/cross scale leaf is scalar-safe."""
    return all(
        _scalar_leaf(s)
        for a in arms
        for s in ([a["intra"][3]] + [c[3] for c in a["cross"]])
    )


def _score_pair(t: Any) -> tuple | None:
    """``matmul(a, b)`` → ``(a, b)``; anything else → ``None``."""
    if isinstance(t, Op) and t.op == "matmul" and len(t.args) == 2:
        return (t.args[0], t.args[1])
    return None


def _dedup(terms: list) -> tuple[list, dict]:
    """First-seen-order unique terms + their index map."""
    seen: dict[Any, int] = {}
    out: list = []
    for t in terms:
        if t not in seen:
            seen[t] = len(out)
            out.append(t)
    return out, seen


# ---------------------------------------------------------------------------
#  Slice plumbing — restacking split/select leaves as views
# ---------------------------------------------------------------------------


def _concrete(shape: Any) -> bool:
    """Return True for a fully-known non-scalar shape tuple."""
    return (
        isinstance(shape, tuple)
        and len(shape) > 0
        and all(isinstance(d, int) for d in shape)
    )


def _slice_index(term: Any) -> tuple | None:
    """Decompose ``select``/``getitem``/``chunk``/``split`` slice ops.

    Returns ``(base, kind, dim, index, nparts, part_size)`` or
    ``None``: ``kind`` is ``"select"`` (indexing removes dim) or
    ``"chunk"`` (equal split keeps dim); ``nparts``/``part_size``
    describe the partition.  ``part_size`` is ``None`` when a split
    cannot be proven equal-size — such slices can't be restacked by
    a reshape.
    """
    if not isinstance(term, Op) or len(term.args) != 1:
        return None
    a = term.attrs
    if term.op == "select":
        dim = a.get("dim")
        idx = a.get("index")
        if isinstance(dim, int) and isinstance(idx, int):
            return (term.args[0], "select", dim, idx, None, None)
        return None
    if term.op == "getitem":
        # t[i] — always indexes dim 0 (the torch_bridge convention).
        idx = a.get("index")
        if isinstance(idx, int):
            return (term.args[0], "select", 0, idx, None, None)
        return None
    if term.op in ("chunk", "split"):
        dim = a.get("dim")
        idx = a.get("index", 0)
        if not isinstance(dim, int) or not isinstance(idx, int):
            return None
        if term.op == "chunk":
            n = a.get("chunks")
            if not isinstance(n, int):
                return None
            base_shape = _shape_of(term.args[0])
            if _concrete(base_shape):
                big = base_shape[dim % len(base_shape)]
                if big % n == 0:
                    return (
                        term.args[0],
                        "chunk",
                        dim,
                        idx,
                        n,
                        big // n,
                    )
            return (term.args[0], "chunk", dim, idx, n, None)
        sizes = a.get("sizes")
        if isinstance(sizes, (list, tuple)) and sizes:
            part = (
                sizes[0] if all(s == sizes[0] for s in sizes) else None
            )
            return (term.args[0], "chunk", dim, idx, len(sizes), part)
        return None
    return None


def _sliced_gather(parts: list) -> tuple | None:
    """Common-base check for a list of ``_slice_index`` results.

    Returns ``(base, kind, dim, nparts, part_size)`` when every leaf
    is the i-th equal slice of the SAME base along the SAME dim, in
    order — so stacking the leaves is one view/permute of the base.
    ``None`` otherwise (caller falls back to per-leaf eval + stack).
    """
    if not parts or any(p is None for p in parts):
        return None
    base0, kind0, dim0, _, n0, part0 = parts[0]
    bshape = _shape_of(base0)
    if not _concrete(bshape):
        return None
    rank = len(bshape)
    dim0n = dim0 % rank
    for i, p in enumerate(parts):
        base, kind, dim, idx, n, part = p
        # ``==`` not ``is``: ``Op.make`` returns the first interned
        # node, whose args hold the original value-equal leaves, so
        # identity is not a stable test for "same base term".
        if base != base0 or kind != kind0 or dim % rank != dim0n:
            return None
        if idx != i:
            return None
        if kind == "chunk" and (
            n != n0 or part is None or part != part0
        ):
            return None
    if kind0 == "select":
        n = bshape[dim0n]
        if n != len(parts):
            return None
        return (base0, "select", dim0, n, 1)
    if n0 != len(parts):
        return None
    return (base0, "chunk", dim0, n0, part0)


def _contig_chunk_range(parts: list) -> tuple | None:
    """Return a contiguous equal-chunk index range of one base.

    ``(base, dim, nparts, part, first_idx, count)`` — the contiguous
    analogue of :func:`_sliced_gather`: every part is a chunk of the
    same base along the same dim with equal part size, indices
    forming one contiguous run.  ``None`` otherwise.
    """
    if not parts or any(p is None for p in parts):
        return None
    base, kind, dim, _, n, part = parts[0]
    if kind != "chunk" or part is None or not _same_chunk_meta(parts):
        return None
    idxs = [p[3] for p in parts]
    if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
        return None
    return (base, dim, n, part, idxs[0], len(idxs))


def _same_chunk_meta(parts: list) -> bool:
    """Return True when every part chunks the same base identically."""
    base, kind, dim, _, n, part = parts[0]
    return all(
        p[0] == base
        and p[1] == kind
        and p[2] == dim
        and p[4] == n
        and p[5] == part
        for p in parts
    )


# ---------------------------------------------------------------------------
#  The plan
# ---------------------------------------------------------------------------


def _state_half(inner: Op) -> tuple:
    """Decompose a cross term's inner ``matmul(A, v)`` for batching.

    Returns ``(k_or_a, u, v, d0, d1)``: when ``A`` is the extracted
    form ``transpose(mul(k, decay_row), d0, d1)``, the decay gate is
    lifted out so it const-caches (``u`` = the ``unsqueeze`` term);
    otherwise ``(A, None, v, None, None)`` and ``A`` evaluates
    through the generic evaluator each forward — still correct.
    """
    a, v = inner.args
    d0, d1 = (
        (a.attrs.get("dim0"), a.attrs.get("dim1"))
        if isinstance(a, Op)
        and a.op == "transpose"
        and len(a.args) == 1
        else (None, None)
    )
    if (
        isinstance(a, Op)
        and isinstance(d0, int)
        and isinstance(d1, int)
        and isinstance(a.args[0], Op)
        and a.args[0].op == "mul"
        and len(a.args[0].args) == 2
    ):
        gk = _gated_factor(a.args[0].args[0], a.args[0].args[1])
        if gk is not None:
            return (gk[1], gk[0], v, d0, d1)
    return (a, None, v, None, None)


def _cross_tables(arms: list[dict]) -> tuple:
    """Dedup the cross operands into q/state tables + pair indices.

    The inner ``matmul(a_j, v_j)`` halves dedup AS TERMS — interned,
    so the pair ``(A_j, v_j)`` is the term's own pairing and never
    re-pairs.
    """
    inners, in_idx = _dedup([c[2] for a in arms for c in a["cross"]])
    q_terms, q_idx = _dedup([c[1] for a in arms for c in a["cross"]])
    pairs = [
        (q_idx[c[1]], in_idx[c[2]], c[0], c[3])
        for a in arms
        for c in a["cross"]
    ]
    return inners, q_terms, pairs


def _uniform_shapes(arms: list[dict], inners: list, q_terms: list):
    """Validate uniform concrete shapes; return the chunk shape."""
    chunk = _uniform([_shape_of(a["intra"][2]) for a in arms])
    score = _uniform([_shape_of(a["intra"][0]) for a in arms])
    if chunk is None or score is None or len(chunk) < 2:
        return None
    if any(
        _uniform([_shape_of(t) for t in ts]) is None
        for ts in (q_terms, inners)
        if ts
    ):
        return None
    return chunk


def _score_operands(arms: list[dict]) -> list | None:
    """Score ``matmul`` operand pairs when every score is one."""
    scores = [_score_pair(a["intra"][0]) for a in arms]
    return scores if all(s is not None for s in scores) else None


def _assemble_plan(arms: list[dict], dim: int) -> dict | None:
    """Validate operand shapes and build the batched schedule."""
    inners, q_terms, pairs = _cross_tables(arms)
    chunk = _uniform_shapes(arms, inners, q_terms)
    if chunk is None or not _scales_safe(arms):
        return None
    states = [_state_half(t) for t in inners]
    return {
        "dim": dim,
        "n": len(arms),
        "chunk": chunk,
        "score_ab": _score_operands(arms),
        "score_terms": [a["intra"][0] for a in arms],
        "decay_terms": [a["intra"][1] for a in arms],
        "v_intra": [a["intra"][2] for a in arms],
        "intra_scale": [a["intra"][3] for a in arms],
        "intra_row": [a["intra"][4] for a in arms],
        "states": states,
        "q_terms": q_terms,
        "pairs": pairs,
        "pair_counts": [len(a["cross"]) for a in arms],
    }


def _plan_of_arms(arm_terms: list, dim: int) -> dict | None:
    """Parse + validate a flat arm list; ``None`` on any failure."""
    parsed = [_parse_arm(a) for a in arm_terms]
    if any(p is None for p in parsed):
        return None
    return _assemble_plan([p for p in parsed if p is not None], dim)


def build_scan_plan(root: Any) -> dict | None:
    """Analyse an ``om_chunk_scan``/``concat`` term into a schedule.

    ``None`` for anything that is not a concat of per-chunk decayed
    sums with uniform shapes — the caller then keeps the spelled
    per-arm evaluation.
    """
    if not (
        isinstance(root, Op) and root.op in ("concat", "om_chunk_scan")
    ):
        return None
    dim = root.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        return None
    return _plan_of_arms(scan_arms(root, dim), dim)


def is_chunk_scan(terms: Any, dim: Any) -> bool:
    """Return True when *terms* flatten to a valid chunked scan.

    *terms* are the concat's bound children — each flattens through
    nested ``concat``/``om_chunk_scan`` nodes on *dim* into leaf
    arms; every leaf must parse as one intra element plus decay-gated
    pair products under uniform concrete shapes.
    """
    if not isinstance(dim, int) or isinstance(dim, bool):
        return False
    arms: list = []
    for t in terms:
        arms += scan_arms(t, dim)
    if len(arms) < 2:
        return False
    return _plan_of_arms(arms, dim) is not None


# ---------------------------------------------------------------------------
#  The scheduled forward
# ---------------------------------------------------------------------------


class ScanRunner:
    """The batched forward for one ``om_chunk_scan`` node.

    Holds the plan and the per-lowering small-tensor caches (pair
    index gathers, per-summand scale rows, broadcast ones) the
    carrier caches per module — keyed ``(tag, device, dtype)`` so a
    moved executor never replays a stale buffer.  ``run(ev)`` is the
    whole forward: *ev* is the executor's operand channel (live
    subterms evaluate once per forward through its per-call memo;
    const subtrees once per lowering through ``_static``).
    """

    def __init__(self, plan: dict) -> None:
        """Bind the plan; caches fill on first forward."""
        self._plan = plan
        self._cache: dict[tuple, Any] = {}

    # -- small-tensor caches -------------------------------------------

    def _cached(
        self, key: tuple, like: Any, make: Any, dtype: Any = None
    ):
        """Device/dtype-aware tensor cache (index tensors, pad rows)."""
        want = dtype or like.dtype
        t = self._cache.get(key)
        if t is None or t.device != like.device or t.dtype != want:
            t = make(like)
            self._cache[key] = t
        return t

    def _scalar(self, t: Any, ev: Any) -> float:
        """Evaluate a scale leaf (Const / Param / literal) to float.

        ``None`` is the peeled ``div``-less summand — an exact 1.0
        divisor, not a missing value.
        """
        torch = _torch()
        if t is None:
            return 1.0
        if isinstance(t, Const):
            return float(t.value)
        if isinstance(t, (int, float)):
            return float(t)
        v = ev(t)
        return float(v.detach() if isinstance(v, torch.Tensor) else v)

    def _scales(self, tag: str, terms: list, ev: Any, like: Any):
        """Per-summand divisors as one cached ``(n,)`` tensor.

        Scale leaves are consts — evaluated once per (device, dtype)
        so steady-state forwards (and CUDA-graph capture) never build
        a host→device tensor.
        """
        torch = _torch()
        key = ("scales", tag, str(like.device), like.dtype)
        t = self._cache.get(key)
        if t is None:
            t = torch.tensor(
                [self._scalar(s, ev) for s in terms],
                dtype=like.dtype,
                device=like.device,
            )
            self._cache[key] = t
        return t

    def _index(self, name: str, idxs: list[int], like: Any):
        """Return a cached long index tensor for the pair gathers."""
        torch = _torch()
        return self._cached(
            ("idx", name, tuple(idxs)),
            like,
            lambda t: torch.tensor(
                idxs, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    # -- stacking helpers ----------------------------------------------

    @staticmethod
    def _pad_row(t: Any, rank: int) -> Any:
        """Left-pad a per-row operand to a chunk's full rank."""
        pad = rank - t.dim()
        if pad > 0:
            t = t.reshape(*((1,) * pad), *t.shape)
        return t

    @staticmethod
    def _pad_stacked(t: Any, rank: int) -> Any:
        """Pad a ``(n, *row)`` stack so each row broadcasts at *rank*."""
        pad = rank - t.dim()
        if pad > 0:
            t = t.reshape(t.shape[0], *((1,) * pad), *t.shape[1:])
        return t

    def _eval_sliced(self, info: tuple, ev: Any) -> Any:
        """Stack ``(n, *leaf_shape)`` from a common-base slice gather.

        ``select`` slices: ``base.movedim(d, 0)``.  ``chunk``/equal
        ``split`` slices: split dim d into ``(n, part)`` by a view,
        then move the block axis front — a permute, zero compute.
        """
        base, kind, dim, n, part = info
        t = ev(base)
        d = dim % t.dim()
        if kind == "select":
            return t.movedim(d, 0)
        t = t.reshape(*t.shape[:d], n, part, *t.shape[d + 1 :])
        return t.movedim(d, 0)

    def _stack_terms(self, terms: list, ev: Any) -> Any:
        """``(n, *leaf)`` stack — a view when terms slice one base.

        Full coverage (the classic ``_sliced_gather`` case) is a
        reshape+permute with no kernel; a *contiguous* index range of
        equal chunks of one base is a reshape + narrow + permute —
        also a view.  Anything else falls back to per-term eval +
        ``torch.stack``.
        """
        torch = _torch()
        parts = [_slice_index(t) for t in terms]
        g = _sliced_gather(parts)
        if g is not None:
            return self._eval_sliced(g, ev)
        rng = _contig_chunk_range(parts)
        if rng is not None:
            base, dim, n, part, i0, cnt = rng
            t = ev(base)
            d = dim % t.dim()
            v = t.reshape(*t.shape[:d], n, part, *t.shape[d + 1 :])
            return v.narrow(d, i0, cnt).movedim(d, 0)
        return torch.stack([ev(t) for t in terms])

    def _row_stack(self, terms: list, like: Any, rank: int, ev: Any):
        """Stack const rows, defaulting a missing gate to ``1``.

        Rows may broadcast differently against the chunk (a ``(c,1)``
        gate, a ``(c,c)`` mask): each pads to *rank*, all expand to
        their common broadcast shape, ``None`` gates contribute an
        exact multiplicative identity.  Callers guarantee ≥1 real
        term — the all-``None`` case never reaches the mul.
        """
        torch = _torch()
        rows = [
            (self._pad_row(ev(t), rank) if t is not None else None)
            for t in terms
        ]
        bt = torch.broadcast_shapes(
            *[r.shape for r in rows if r is not None]
        )
        one = self._cached(
            ("ones", bt),
            like,
            lambda t: torch.ones(bt, dtype=t.dtype, device=t.device),
        )
        return torch.stack(
            [r.expand(bt) if r is not None else one for r in rows]
        )

    # -- the two halves -------------------------------------------------

    def _forward_intra(self, ev: Any) -> Any:
        """Compute the stacked ``(s ⊙ d) @ v / S`` intra-chunk product."""
        torch = _torch()
        plan = self._plan
        sab = plan["score_ab"]
        if sab is not None:
            s = torch.matmul(
                self._stack_terms([p[0] for p in sab], ev),
                self._stack_terms([p[1] for p in sab], ev),
            )
        else:
            s = self._stack_terms(plan["score_terms"], ev)
        dts = plan["decay_terms"]
        if all(t == dts[0] for t in dts):
            # A shared decay mask broadcasts — no K-way copy.
            d = self._pad_row(ev(dts[0]), s.dim() - 1).unsqueeze(0)
        else:
            d = self._pad_stacked(
                torch.stack([ev(t) for t in dts]), s.dim()
            )
        v = self._stack_terms(plan["v_intra"], ev)
        out = torch.matmul(s * d, v)  # (K, *batch, c, dv)
        if any(r is not None for r in plan["intra_row"]):
            rows = self._row_stack(
                plan["intra_row"], out, len(plan["chunk"]), ev
            )
            out = rows * out
        sc = self._scales("intra", plan["intra_scale"], ev, out)
        return out / sc.reshape(-1, *((1,) * (out.dim() - 1)))

    def _state_row(self, s: tuple, ev: Any) -> Any:
        """One cross inner lhs — eval-mode or the parsed gated key."""
        kk = ev(s[0])
        if s[1] is None:
            return kk
        kd = kk * self._pad_row(ev(s[1]), kk.dim())
        return kd.transpose(s[3], s[4])

    def _state_stack(self, states: list, ev: Any) -> Any:
        """Stack the ``(J, …)`` cross inner lhs terms.

        When every inner is the uniform ``transpose(k·U)`` form the
        decay product batches over the stack axis and the transpose
        applies once; otherwise per-term eval.
        """
        torch = _torch()
        if all(s[1] is not None for s in states) and all(
            (s[3], s[4]) == (states[0][3], states[0][4]) for s in states
        ):
            ks = self._stack_terms([s[0] for s in states], ev)
            us = self._row_stack(
                [s[1] for s in states], ks, ks.dim() - 1, ev
            )
            kd = ks * us
            r = kd.dim() - 1
            d0, d1 = states[0][3] % r + 1, states[0][4] % r + 1
            return kd.transpose(d0, d1)
        return torch.stack([self._state_row(s, ev) for s in states])

    def _forward_cross(self, ev: Any) -> Any | None:
        """Compute the batched ``Σ_j row_ij ⊙ (q_i @ M_j) / S`` terms."""
        torch = _torch()
        plan = self._plan
        pairs = plan["pairs"]
        if not pairs:
            return None
        states = plan["states"]
        a = self._state_stack(states, ev)
        vc = self._stack_terms([s[2] for s in states], ev)
        m = torch.matmul(a, vc)  # (J, *batch, e, dv)
        q = self._stack_terms(plan["q_terms"], ev)
        qi = self._index("q", [p[0] for p in pairs], q)
        mi = self._index("m", [p[1] for p in pairs], q)
        prod = torch.matmul(
            q.index_select(0, qi), m.index_select(0, mi)
        )  # (P, *batch, c, dv)
        crank = len(plan["chunk"])
        psc = self._scales("cross", [p[3] for p in pairs], ev, prod)
        if any(p[2] is not None for p in pairs):
            prod = prod * (
                self._row_stack([p[2] for p in pairs], prod, crank, ev)
                / psc.reshape(-1, *((1,) * crank))
            )
        else:
            prod = prod / psc.reshape(-1, *((1,) * crank))
        # Deterministic accumulation: pairs are contiguous per target
        # chunk (arm order), so segment sums replace a scatter-add.
        parts = prod.split(plan["pair_counts"])
        return torch.stack([p.sum(0) for p in parts])

    def run(self, ev: Any) -> Any:
        """Run the pairwise-batched schedule over operand channel *ev*."""
        plan = self._plan
        out = self._forward_intra(ev)
        cross = self._forward_cross(ev)
        if cross is not None:
            out = out + cross
        # Assemble: (K, *chunk) → concat along the term's own dim.
        dn = plan["dim"] % len(plan["chunk"])
        return out.movedim(0, dn).flatten(dn, dn + 1)
