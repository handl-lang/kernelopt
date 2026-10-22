# ruff: noqa: RUF002, RUF003 — math notation (≡, σ, −, ·) in docstrings.
"""``KernelSink`` — the kernel codegen backend behind the Sink port.

A peer of ``mlopt_onnx.OnnxSink``, smaller: the backend is a *torch
composite*.  The binding table is a deliberately small op set: the
elementwise + reduction surface the codegen law bank can reach, the
layout ops real extracted carriers carry (``split``/``concat``/
``contiguous``/``detach_`` — views or copies, never fusion sites),
the ``linear`` GEMM+epilogue slot, and the decayed-scan vocabulary
(``decay``/``decay_vec``/``om_decayed``) the retnet-class carriers
need.  Each fused op (``FUSED_KERNEL_OPS``) lowers to
``torch.compile`` of its composite body — one fused kernel
(Inductor-generated, Triton on CUDA) where the spelled program
launched an op at a time.

Verify-first, per the ONNX precedent: :meth:`KernelSink.verify`
runs the spelled reference and the scheduled candidate in *this*
sink's runtime and returns the equivalence report — a fused kernel
ships only when it is numerically equal, never on the law's
say-so.  ``torch`` is imported lazily: the package's hard
dependency is ``mlopt-core`` alone, matching the sink packages'
adapter layering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.ops import OpTable
from mlopt_core.ports import ExecutorSpec, OpRegistry

from mlopt_kernel.laws import FUSED_KERNEL_OPS, SCHEDULED_KERNEL_OPS
from mlopt_kernel.scan import ScanRunner, build_scan_plan

if TYPE_CHECKING:
    from mlopt_core.ir import IR

__all__ = [
    "KernelExecutor",
    "KernelSink",
    "KernelVerifyReport",
    "kernel_ops",
]


def _torch() -> Any:
    """Return torch lazily — the sink's only third-party reach."""
    import torch

    return torch


def _split_sizes(sizes: Any) -> Any:
    """``split(x, sizes_list)`` and ``split(x, int)`` — one attr name.

    A non-empty list/tuple is the explicit per-part sizes; anything
    else is the equal-chunk width (the ``torch_bridge`` convention).
    """
    if isinstance(sizes, (list, tuple)) and sizes:
        return list(sizes)
    return int(sizes if isinstance(sizes, int) else 1)


def _decay(g: Any, t: int = 0) -> Any:
    """``tril(γ^{i−j})`` on the last two axes — the ``decay`` generator.

    The carrier semantics, vendored (the kernel package stays
    carrier-free): γ's dims are LEADING (batch) dims, so a scalar
    rate gives ``(T,T)`` and a per-head ``(H,)`` rate ``(H,T,T)``.
    ``torch.as_tensor`` normalises Python-scalar leaves.
    """
    torch = _torch()
    t = int(t or 0)
    g = torch.as_tensor(g)
    pos = torch.arange(t, dtype=g.dtype, device=g.device)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)
    return torch.tril(g[..., None, None] ** diff.clamp_min(0))


def _decay_vec(
    g: Any, t: int = 0, offset: int = 0, sign: int = 1
) -> Any:
    """``g^(sign·k + offset)``, ``k`` in ``[0,T)`` — ``decay_vec``.

    Same leading-batch-dims convention as :func:`_decay`: a scalar
    rate gives ``(T,)``, a per-head ``(H,)`` rate ``(H,T)``.
    ``sign=0`` is a legal exponent (``g^offset``, a constant row),
    so the attr is read with an explicit default, never ``or``.
    """
    torch = _torch()
    t = int(t or 0)
    off = int(offset or 0)
    g = torch.as_tensor(g)
    pos = torch.arange(t, dtype=g.dtype, device=g.device)
    return g[..., None] ** (pos * int(sign) + off)


def _om_decayed(s: Any, d: Any, v: Any) -> Any:
    """``(s ⊙ d) @ v`` — the fused decayed-attention composite body.

    The decayed carrier's own spelled semantics: the masked score
    matmul the chunked decayed scan's intra-chunk element computes.
    In the fused-op table it compiles as one unit — the elementwise
    decay product folds into the contraction instead of
    materialising ``(…,T,K)``.
    """
    torch = _torch()
    return torch.matmul(s * d, v)


def _kf_layernorm(u: Any, dim: Any, eps: float) -> Any:
    """``(u−μ)·rsqrt(mean((u−μ)²)+eps)`` — the fused-layernorm body.

    ``dim`` is the LHS ``mean``'s reduce spec verbatim (int or tuple)
    — the composite is *defined* as the spelled math, so whatever
    axes the site reduces, this body recomputes.
    """
    torch = _torch()
    c = u - torch.mean(u, dim=dim, keepdim=True)
    return c * torch.rsqrt(
        torch.mean(c.pow(2), dim=dim, keepdim=True) + eps
    )


def _kf_rmsnorm(u: Any, dim: Any, eps: float) -> Any:
    """``u·rsqrt(mean(u²)+eps)`` — the gain-free fused-rmsnorm body.

    Same contract as ``_kf_layernorm``: ``dim`` is the site's
    ``mean`` reduce spec verbatim.
    """
    torch = _torch()
    return u * torch.rsqrt(
        torch.mean(u.pow(2), dim=dim, keepdim=True) + eps
    )


def _kf_sdpa(
    a: Any, w: Any, s: Any, v: Any, dim: Any, cap: bool = False
) -> Any:
    """``softmax(mm(a,w)·s)·v`` — the whole-attention composite body.

    One compiled unit covering the qk matmul, the scale, the
    (max-shifted) softmax and the value matmul — where the
    ``kf_softmax_mm(kf_mm_scale(...), v)`` chain launched two kernels
    with the scaled-score buffer materialised between them.
    ``cap=True`` applies the ``tanh`` soft-cap to the scaled score —
    the bounded-logit spelling (the spelled reference stays finite
    in fp32 where an uncapped ``exp(qk)`` can overflow at scale).
    """
    torch = _torch()
    score = torch.matmul(a, w) * s
    if cap:
        score = torch.tanh(score)
    return torch.matmul(torch.softmax(score, dim=dim), v)


def _bindings() -> dict[str, Any]:
    """Build the kernel op table — a small torch composite vocabulary.

    Covers the elementwise surface plus the reduction/view ops the
    spelled fixtures reach, and the ``kf_*`` fused-kernel ops the law
    bank mints — each bound to its *composite body* (what the fused
    kernel computes; the executor compiles it into one launch).
    Args arrive evaluated (tensor or scalar); attrs come through as
    canonical kwargs.
    """
    torch = _torch()

    return {
        "abs": torch.abs,
        "add": torch.add,
        "atan2": torch.atan2,
        "ceil": torch.ceil,
        "clamp": torch.clamp,
        # --- layout ops: materialise or view, never a fusion site
        "concat": lambda *ts, dim=0: torch.cat(list(ts), dim=int(dim)),
        "contiguous": lambda x: x.contiguous(),
        "cos": torch.cos,
        "cosh": torch.cosh,
        # --- the decayed-scan generators (const-cached by the
        # executor's static-subtree memo — see KernelExecutor)
        "decay": _decay,
        "decay_vec": _decay_vec,
        "detach": lambda t: t.detach(),
        "detach_": lambda t: t.detach(),
        "div": torch.div,
        "elu": torch.nn.functional.elu,
        "erf": torch.erf,
        "exp": torch.exp,
        "exp2": torch.exp2,
        "expm1": torch.expm1,
        "flatten": lambda x, start_dim=0, end_dim=-1: torch.flatten(
            x, start_dim, end_dim
        ),
        "floor": torch.floor,
        "fmax": torch.fmax,
        "fmin": torch.fmin,
        "fmod": torch.fmod,
        "gelu": torch.nn.functional.gelu,
        "hardsigmoid": torch.nn.functional.hardsigmoid,
        "hardswish": torch.nn.functional.hardswish,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "linear": lambda x, w, *a, **kw: torch.nn.functional.linear(
            x, w, a[0] if a else kw.get("bias")
        ),
        "log": torch.log,
        "log10": torch.log10,
        "log1p": torch.log1p,
        "log2": torch.log2,
        "matmul": torch.matmul,
        "maximum": torch.maximum,
        "mean": lambda x, dim=None, keepdim=False: torch.mean(
            x, dim=dim, keepdim=keepdim
        ),
        "minimum": torch.minimum,
        "mish": torch.nn.functional.mish,
        "mul": torch.mul,
        # ``neg``/``reciprocal`` lift scalar args (a ``Const`` leaf
        # evaluates to a python float) to a 0-d tensor — the binding
        # contract is "tensor or scalar", and the unary aten ops only
        # take tensors.
        "neg": lambda t: torch.neg(torch.as_tensor(t)),
        "om_decayed": _om_decayed,
        # The scheduled-op spelled body: ``om_chunk_scan`` *is* the
        # concat of its arms — the verify-first fallback (and what a
        # non-plan node lowers to); the scheduled path lives in the
        # executor's ScanRunner dispatch, not in this table.
        "om_chunk_scan": lambda *ts, dim=0: torch.cat(
            list(ts), dim=int(dim)
        ),
        "permute": lambda x, dims: x.permute(*dims),
        "pow": torch.pow,
        "reciprocal": lambda t: torch.reciprocal(torch.as_tensor(t)),
        "relu": torch.relu,
        "remainder": torch.remainder,
        "reshape": lambda x, shape: (
            x.reshape(*shape)
            if isinstance(shape, (tuple, list))
            else x.reshape(shape)
        ),
        "round": torch.round,
        "rsqrt": torch.rsqrt,
        "select": lambda x, dim, index: x.select(dim, index),
        "sigmoid": torch.sigmoid,
        "sign": torch.sign,
        "silu": torch.nn.functional.silu,
        "sin": torch.sin,
        "sinh": torch.sinh,
        "softmax": lambda x, dim=-1: torch.softmax(x, dim=dim),
        "softplus": torch.nn.functional.softplus,
        "split": lambda t, sizes=(), dim=-1, index=0: torch.split(
            t, _split_sizes(sizes), dim=int(dim)
        )[int(index)],
        "sqrt": torch.sqrt,
        "square": torch.square,
        "squeeze": lambda x, dim=None: (
            x.squeeze() if dim is None else x.squeeze(dim)
        ),
        "sub": torch.sub,
        "sum": lambda x, dim=None, keepdim=False: torch.sum(
            x, dim=dim, keepdim=keepdim
        ),
        "tanh": torch.tanh,
        "transpose": lambda x, dim0=0, dim1=1: x.transpose(dim0, dim1),
        "unsqueeze": lambda x, dim: x.unsqueeze(dim),
        # --- the kernel vocabulary: composite bodies, one launch each
        "kf_silu": lambda a, b: a * torch.sigmoid(b),
        "kf_softsign": lambda a, b, c: a / (b.abs() + c),
        "kf_addmul": lambda a, b, c: a * b + c,
        "kf_swiglu": lambda a, b: a * torch.sigmoid(a) * b,
        "kf_silu_sigmoid": lambda a, b: (
            a * torch.sigmoid(a) * torch.sigmoid(b)
        ),
        "kf_addmul_tanh": lambda a, b, c: a * b + torch.tanh(c),
        "kf_addmul2": lambda a, b, c, d: a * b + c * d,
        "kf_mul3": lambda a, b, c: a * b * c,
        "kf_softmax": lambda u, dim: torch.softmax(u, dim=dim),
        "kf_softmax_mm": lambda u, v, dim: torch.matmul(
            torch.softmax(u, dim=dim), v
        ),
        "kf_sdpa": _kf_sdpa,
        "kf_layernorm": _kf_layernorm,
        "kf_rmsnorm": _kf_rmsnorm,
        "kf_mm_bias": lambda a, w, b: torch.matmul(a, w) + b,
        "kf_mm_scale": lambda a, w, s: torch.matmul(a, w) * s,
        "kf_mm_sb": lambda a, w, s, b: torch.matmul(a, w) * s + b,
        "kf_mm_gelu": lambda a, w: torch.nn.functional.gelu(
            torch.matmul(a, w)
        ),
    }


#: The binding table, built once — the lambdas are stateless.
_BINDINGS: dict[str, Any] | None = None


def kernel_ops() -> frozenset[str]:
    """Return the op names the sink can lower — ``supported_ops``."""
    global _BINDINGS
    if _BINDINGS is None:
        _BINDINGS = _bindings()
    return frozenset(_BINDINGS)


def _table() -> dict[str, Any]:
    """Return the shared binding dict — one build, every consumer."""
    kernel_ops()
    if _BINDINGS is None:
        raise RuntimeError("binding table not built")
    return _BINDINGS


@dataclass(frozen=True)
class KernelVerifyReport:
    """Hold the sink's equivalence report — a ``VerifyResult``."""

    max_abs: float
    max_rel: float
    passed: bool


_MISSING = object()


class KernelExecutor:
    """The lowered artifact — a callable executor over an IR term.

    Evaluates the term against the sink's binding table; each fused
    op (``FUSED_KERNEL_OPS``) dispatches through a
    ``torch.compile``-compiled callable (cached per op) — Inductor
    fuses the composite body into a single kernel launch where the
    spelled ops launched one at a time.  With ``compile_fused=False``
    the fused ops run their spelled bodies eagerly — the fallback
    verify-first compares against.

    *Static-subtree memo*: a subtree with no ``Var`` leaf is
    input-independent — its value is fixed at ``lower`` time for the
    executor's lifetime.  The evaluator computes it once and caches
    it (``_static``), so the ``decay``/``decay_vec`` mask generators
    and any layout-only slices of them leave the steady-state loop
    entirely — the same schedule choice the carrier executor's
    ``_const`` cache makes, and the difference between a sink that
    materialises its constants per forward and one that does not.

    *Per-forward memo*: a live op's value is cached for the call's
    duration (``_memo``, reset in :meth:`forward`), so a DAG-shared
    subterm — the retnet carrier's shared q/k/v projections feeding
    every chunk's ``split`` — evaluates once per forward, not once
    per use.

    *Scheduled ops* (``SCHEDULED_KERNEL_OPS``): ``om_chunk_scan``
    marks a ``concat``-of-chunks the laws folded.  Under
    ``compile_fused`` the node dispatches to a
    :class:`~mlopt_kernel.scan.ScanRunner` — the pairwise-batched
    chunk schedule (batched GEMMs + segment sums) instead of the
    per-op dispatch tape; when the term doesn't re-parse as a scan
    the node falls back to its spelled ``cat`` body — never a silent
    drop.  ``compile_fused=False`` always runs the spelled body.
    """

    def __init__(
        self,
        ir: IR,
        params: Mapping[str, Any] | None = None,
        *,
        compile_fused: bool = True,
        bindings: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind the program, its leaf values, and the codegen flag."""
        self._ir = ir
        self._params = dict(params or {})
        self._bindings = (
            dict(bindings) if bindings is not None else dict(_table())
        )
        self.compile_fused = compile_fused
        self._compiled: dict[str, Any] = {}
        self._live: dict[int, bool] = {}
        self._mark_live(ir.root)
        self._static: dict[int, Any] = {}
        self._memo: dict[int, Any] = {}
        self._scans: dict[int, Any] = {}

    def forward(self, *xs: Any) -> Any:
        """Run the program on positional inputs in ``ir.inputs`` order."""
        env = dict(self._params)
        env.update(
            {
                v.name: x
                for v, x in zip(self._ir.inputs, xs, strict=True)
            }
        )
        self._memo = {}
        return self._eval(self._ir.root, env)

    def __call__(self, *xs: Any) -> Any:
        """Alias of :meth:`forward` — the callable executor surface."""
        return self.forward(*xs)

    def _mark_live(self, t: Any) -> bool:
        """Flag every Op node whose subtree reaches a ``Var`` leaf."""
        if isinstance(t, Var):
            return True
        if isinstance(t, Op):
            k = id(t)
            if k not in self._live:
                self._live[k] = any(self._mark_live(a) for a in t.args)
            return self._live[k]
        return False

    def _leaf(self, t: Any, env: dict) -> Any:
        """Resolve a leaf: Var/Param by name, Const by value."""
        if isinstance(t, Const):
            return t.value
        if isinstance(t, (Var, Param)):
            return env[t.name]
        raise TypeError(f"unsupported leaf term {t!r}")

    def _fused(self, op: str) -> Any:
        """Return the compiled callable for one fused op (cached)."""
        fn = self._compiled.get(op)
        if fn is None:
            raw = self._bindings[op]
            fn = (
                _torch().compile(raw, fullgraph=False)
                if self.compile_fused
                else raw
            )
            self._compiled[op] = fn
        return fn

    def _scheduled(self, t: Op, env: dict) -> Any:
        """Run *t* through its batched plan (or the spelled fallback)."""
        key = id(t)
        runner = self._scans.get(key, _MISSING)
        if runner is _MISSING:
            plan = build_scan_plan(t)
            runner = None if plan is None else ScanRunner(plan)
            self._scans[key] = runner
        if runner is not None:
            return runner.run(lambda u: self._eval(u, env))
        # Not a scan after all — the spelled ``cat`` body, honestly.
        fn = self._bindings[t.op]
        return fn(
            *[self._eval(a, env) for a in t.args], **dict(t.attrs)
        )

    def _apply(self, t: Op, env: dict) -> Any:
        """Dispatch one op node — fused ops through the compiled path."""
        if t.op in SCHEDULED_KERNEL_OPS and self.compile_fused:
            return self._scheduled(t, env)
        fn = (
            self._fused(t.op)
            if t.op in FUSED_KERNEL_OPS
            else self._bindings.get(t.op)
        )
        if fn is None:
            raise KeyError(
                f"KernelSink cannot lower op {t.op!r} — not in "
                "supported_ops"
            )
        return fn(
            *[self._eval(a, env) for a in t.args], **dict(t.attrs)
        )

    def _eval(self, t: Any, env: dict) -> Any:
        """Evaluate *t* — static subtrees once, the rest per call."""
        if not isinstance(t, Op):
            return self._leaf(t, env)
        k = id(t)
        if self._live.get(k, True):
            hit = self._memo.get(k, _MISSING)
            if hit is _MISSING:
                hit = self._apply(t, env)
                self._memo[k] = hit
            return hit
        if k not in self._static:
            self._static[k] = self._apply(t, env)
        return self._static[k]


class KernelSink:
    """``mlopt_core.ports.Sink`` over the kernel composite backend.

    ``supported_ops`` bounds the search the same way ``OnnxSink``'s
    does — any member using an op outside the table prices at ``+inf``
    under ``backend_cost``, so the search never commits to a form this
    sink cannot execute.
    """

    def __init__(self, *, compile_fused: bool = True) -> None:
        """Configure the sink; ``compile_fused`` toggles the codegen."""
        self.compile_fused = compile_fused
        self._ops: OpTable | None = None

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every op name this sink lowers — the extraction bound."""
        return kernel_ops()

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry — an ``OpTable`` of torch bindings."""
        if self._ops is None:
            table = OpTable.core()
            table.torch_bindings = dict(_table())
            self._ops = table
        return self._ops

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """Empty — one lowering, no carrier routing (OnnxSink's shape)."""
        return {}

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> KernelExecutor:
        """Materialise ``ir`` as a runnable kernel executor."""
        return KernelExecutor(
            ir, params, compile_fused=self.compile_fused
        )

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> KernelVerifyReport:
        """Run ``ref`` and ``opt`` on ``inputs``; compare in this runtime.

        ``ref`` may be any callable (the spelled torch module, or a
        second executor); ``opt`` is the lowered candidate.  The
        metric is the repo's rel-diff gate: ``max|ref−out| /
        (max|ref| + 1e-8) ≤ rtol`` — or ``max_abs ≤ atol`` when the
        caller supplies an absolute floor.
        """
        torch = _torch()
        del torch  # comparison is tensor arithmetic, not torch calls
        args = (
            inputs if isinstance(inputs, (tuple, list)) else (inputs,)
        )
        ref_out = ref(*args)
        out = opt(*args)
        diff = (ref_out - out).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        denom = (
            float(ref_out.abs().max()) + 1e-8
            if ref_out.numel()
            else 1e-8
        )
        max_rel = max_abs / denom
        passed = max_rel <= rtol or (
            atol is not None and max_abs <= atol
        )
        return KernelVerifyReport(
            max_abs=max_abs, max_rel=max_rel, passed=bool(passed)
        )
