"""The kernelopt spike's bench cases — measured codegen pay.

Twenty-two kernel cases: a spelled program, a codegen-law candidate, and
the honest row — lower both through
:class:`mlopt_kernel.KernelSink`, verify numerically (verify-first),
then time spelled vs scheduled.  Where the case has a registered
hand-Triton kernel (``bench.limits.triton``), the row times and
verifies that arm too — the practitioner's kernel is the field
baseline the fused composite is measured against.  This is the
spike's measured half of the go/no-go: does the engine emit a
verified-equal kernel at lower measured cost than the spelled
baseline.

Run::

    .venv/bin/python -m bench.discovery.kernel
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, TensorType, Var

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost

#: (name, category, spelled builder, rules to fire in order)
CASES: tuple[tuple[str, str, Any, tuple[str, ...]], ...] = ()


def _v(name, shape):
    return Var(name, TensorType(tuple(shape)))


def _mk():
    x = _v("x", (4096, 512))
    y = _v("y", (4096, 512))
    w = _v("w", (512, 512))
    b = _v("b", (512,))
    return x, y, w, b


def _mk_decayed():
    s = _v("s", (16, 128, 128))
    d = _v("d", (128, 128))
    v = _v("v", (16, 128, 128))
    return s, d, v


def _mk_decayed_cases():
    """The carrier-style decayed-scan case builders — real spellings.

    The ``om_decayed`` vocabulary measured against the spellings the
    chunked decayed scan actually emits: the generated ``decay`` mask,
    the ``linear``-produced score, the ``decay_vec``-gated cross term,
    and the ``split``-chunked pair.  Every decay rate is a live
    ``sigmoid(g)`` feed — the carrier parametrisation (g a learned
    scalar, not a baked literal: a ``Const`` rate would pin the mask
    generator to CPU and break under CUDA feeds).
    """
    # the generated-mask element: (s * tril(g^(i-j))) @ v
    sm = _v("sm", (16, 128, 128))
    gm = _v("gm", ())
    vm = _v("vm", (16, 128, 128))
    mask = Op.make("decay", Op.make("sigmoid", gm), t=128)
    # the linear-projected score: (linear(x,w) ⊙ decay) @ v
    xi = _v("xi", (128, 64))
    wi = _v("wi", (128, 64))
    gi = _v("gi", ())
    vi = _v("vi", (128, 64))
    # the two-term block arm: intra element + decay_vec cross term
    sp = _v("sp", (128, 128))
    gp = _v("gp", ())
    vp = _v("vp", (128, 64))
    gr = _v("gr", ())
    xp = _v("xp", (128, 64))
    wp = _v("wp", (64, 64))
    up = _v("up", (64, 64))
    row_gate = Op.make(
        "unsqueeze",
        Op.make("decay_vec", Op.make("sigmoid", gr), t=128),
        dim=-1,
    )
    # the two-chunk scan: split views, per-chunk om_decayed, summed
    sc = _v("sc", (16, 128, 64))
    gc = _v("gc", ())
    vc = _v("vc", (16, 128, 64))
    chunk_decay = Op.make("decay", Op.make("sigmoid", gc), t=64)

    def chunk(i):
        return Op.make(
            "matmul",
            Op.make(
                "mul",
                Op.make("split", sc, sizes=(64, 64), dim=1, index=i),
                chunk_decay,
            ),
            Op.make("split", vc, sizes=(64, 64), dim=1, index=i),
        )

    return (
        (
            "kernel_decayed_mask",
            "speed",
            Op.make("matmul", Op.make("mul", sm, mask), vm),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_intra",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "mul",
                    Op.make("linear", xi, wi),
                    Op.make("decay", Op.make("sigmoid", gi), t=128),
                ),
                vi,
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_pair",
            "speed",
            Op.make(
                "add",
                Op.make(
                    "matmul",
                    Op.make(
                        "mul",
                        sp,
                        Op.make("decay", Op.make("sigmoid", gp), t=128),
                    ),
                    vp,
                ),
                Op.make(
                    "mul",
                    row_gate,
                    Op.make("matmul", xp, Op.make("matmul", wp, up)),
                ),
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_decayed_chunks",
            "speed",
            Op.make("add", chunk(0), chunk(1)),
            ("kern_fuse_om_decayed",),
        ),
    )


def _cases():
    x, y, w, b = _mk()
    s, d, v = _mk_decayed()
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    return (
        (
            "kernel_silu",
            "speed",
            Op.make("mul", x, Op.make("sigmoid", x)),
            ("kern_fuse_silu",),
        ),
        (
            "kernel_softsign",
            "speed",
            Op.make(
                "div", x, Op.make("add", Op.make("abs", x), Const(1))
            ),
            ("kern_fuse_softsign",),
        ),
        (
            "kernel_fma",
            "speed",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
            ("kern_fuse_addmul", "kern_fuse_addmul_tanh"),
        ),
        (
            "kernel_swiglu",
            "speed",
            Op.make("mul", Op.make("mul", x, Op.make("sigmoid", x)), y),
            ("kern_fuse_silu", "kern_fuse_swiglu"),
        ),
        (
            "kernel_softmax",
            "speed",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=-1, keepdim=True),
            ),
            ("kern_fuse_softmax",),
        ),
        (
            "kernel_layernorm",
            "speed",
            Op.make(
                "mul",
                ln_center,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", ln_center, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-5),
                    ),
                ),
            ),
            ("kern_fuse_layernorm",),
        ),
        (
            "kernel_mm_bias",
            "speed",
            Op.make("add", Op.make("matmul", x, w), b),
            ("kern_fuse_mm_bias",),
        ),
        (
            "kernel_mm_scale_bias",
            "speed",
            Op.make(
                "add",
                Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                b,
            ),
            ("kern_fuse_mm_scale_bias", "kern_fuse_addmul"),
        ),
        (
            "kernel_mm_scale",
            "speed",
            Op.make("mul", Op.make("matmul", x, w), Const(0.125)),
            ("kern_fuse_mm_scale",),
        ),
        (
            "kernel_mm_gelu",
            "speed",
            Op.make("gelu", Op.make("matmul", x, w)),
            ("kern_fuse_mm_gelu",),
        ),
        (
            "kernel_rmsnorm",
            "speed",
            Op.make(
                "mul",
                x,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", x, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-6),
                    ),
                ),
            ),
            ("kern_fuse_rmsnorm",),
        ),
        # The bounded-logit attention chain — ``softmax(tanh(qk))·v``,
        # the soft-cap spelling.  The un-capped ``softmax(qk)`` spelled
        # site overflows fp32 at bench scale (exp(qk) → inf where the
        # fused max-shifted softmax stays finite) — a documented
        # *stability* win for the fused op, kept out of the timed row
        # so the verify gate compares finite references.
        (
            "kernel_softmax_mm",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make("tanh", Op.make("matmul", x, w)),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make("tanh", Op.make("matmul", x, w)),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            ("kern_fuse_softmax", "kern_fuse_softmax_mm"),
        ),
        (
            "kernel_sdpa",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "mul",
                            Op.make("matmul", x, w),
                            Const(0.25),
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "mul",
                                Op.make("matmul", x, w),
                                Const(0.25),
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa",
            ),
        ),
        # The soft-capped twin — ``softmax(tanh(mm·s))·v``, the
        # Gemma-2 bounded-logit spelling.  ``kern_fuse_sdpa_cap``
        # folds the whole block to one ``kf_sdpa(cap=True)`` launch;
        # the chain laws compete as the two-launch fallback.
        (
            "kernel_sdpa_cap",
            "speed",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make(
                        "exp",
                        Op.make(
                            "tanh",
                            Op.make(
                                "mul",
                                Op.make("matmul", x, w),
                                Const(0.25),
                            ),
                        ),
                    ),
                    Op.make(
                        "sum",
                        Op.make(
                            "exp",
                            Op.make(
                                "tanh",
                                Op.make(
                                    "mul",
                                    Op.make("matmul", x, w),
                                    Const(0.25),
                                ),
                            ),
                        ),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                w,
            ),
            (
                "kern_fuse_mm_scale",
                "kern_fuse_softmax",
                "kern_fuse_softmax_mm",
                "kern_fuse_sdpa_cap",
            ),
        ),
        (
            "kernel_decayed",
            "speed",
            Op.make(
                "matmul",
                Op.make("mul", s, d),
                v,
            ),
            ("kern_fuse_om_decayed",),
        ),
        (
            "kernel_strength",
            "speed",
            Op.make(
                "add",
                Op.make("pow", x, Const(2)),
                Op.make("div", y, Const(4.0)),
            ),
            (
                "kern_square_strength",
                "kern_div_recip",
                "kern_fuse_addmul",
            ),
        ),
        (
            "kernel_mixed",
            "memory",
            Op.make(
                "mul",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("sigmoid", y),
            ),
            (
                "kern_fuse_silu",
                "kern_fuse_swiglu",
                "kern_fuse_silu_sigmoid",
            ),
        ),
        (
            "kernel_flat",
            "speed",
            Op.make("add", Op.make("tanh", x), Op.make("cos", y)),
            (),
        ),
        *_mk_decayed_cases(),
    )


@dataclass
class KernelRow:
    """One measured case row — spelled vs scheduled, verified."""

    name: str
    category: str
    scheduled: str = ""
    verified: bool | None = None
    max_abs: float | None = None
    cost_before: float = 0.0
    cost_after: float = 0.0
    spelled_us: float = 0.0
    scheduled_us: float = 0.0
    speedup: float | None = None
    triton_us: float | None = None
    triton_verified: bool | None = None
    triton_note: str = ""
    detail: dict = field(default_factory=dict)


def _time(fn, args, reps=50, device=None):
    import torch

    args = tuple(a.to(device) for a in args) if device else args
    for _ in range(10):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _vars_of(t):
    out = []

    def walk(u):
        if isinstance(u, Var):
            if u.name not in [v.name for v in out]:
                out.append(u)
        elif isinstance(u, Op):
            for a in u.args:
                walk(a)

    walk(t)
    return out


def _triton_arm(name):
    """Return ``(callable, None)`` for case *name*'s hand Triton kernel.

    The honest skip cases: triton not installed, no CUDA, or no
    kernel registered — the row records the reason, never a silent
    drop.
    """
    if importlib.util.find_spec("triton") is None:
        return None, "triton not installed"
    import torch

    if not torch.cuda.is_available():
        return None, "needs CUDA"
    from bench.limits.triton import kernel_builder

    mk = kernel_builder(name)
    if mk is None:
        return None, f"no triton kernel registered for {name!r}"
    return mk(), None


def run_case(
    name, category, term, rules_to_fire, *, device=None, reps=50
):
    """Board the spelled term, fire the laws, verify + time delivery."""
    import torch

    row = KernelRow(name=name, category=category)
    eg = EGraph()
    root = eg.add_term(term)
    for rn in rules_to_fire:
        r = next(r for r in KERNEL_RULES if r.name == rn)
        eg.apply_rule(r, root)
    best = eg.extract_best(root, kernel_cost) or term
    row.scheduled = repr(best)
    row.cost_before = kernel_cost(term)
    row.cost_after = kernel_cost(best)

    inputs = _vars_of(term)
    params = {}
    ir_s = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    )
    ir_b = IR(root=best, inputs=inputs, input_names=ir_s.input_names)
    sink = KernelSink()
    spelled = sink.lower(ir_s)
    sched = sink.lower(ir_b)
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float32)
        for v in inputs
    )
    rep = sink.verify(spelled, sched, feeds)
    row.verified = bool(rep.passed)
    row.max_abs = rep.max_abs
    row.spelled_us = _time(spelled, feeds, reps, device)
    row.scheduled_us = _time(sched, feeds, reps, device)
    if row.scheduled_us:
        row.speedup = round(row.spelled_us / row.scheduled_us, 3)

    arm, why = _triton_arm(name)
    if arm is None:
        row.triton_note = why or "unavailable"
    else:
        try:
            feeds_dev = tuple(a.to(device) for a in feeds)
            rep_t = sink.verify(spelled, arm, feeds_dev)
            row.triton_verified = bool(rep_t.passed)
            row.triton_us = _time(arm, feeds_dev, reps, device)
        except Exception as e:  # kernel failed — honest row
            row.triton_note = f"triton arm: {e}"
    return row


def main() -> list[KernelRow]:
    """Run every kernel case; print the honest row."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name, cat, term, fires in _cases():
        row = run_case(name, cat, term, fires, device=device)
        rows.append(row)
        tri = (
            f"triton {row.triton_us:.1f}us (verified={row.triton_verified})"
            if row.triton_us is not None
            else f"triton — ({row.triton_note})"
        )
        print(  # stdout-compat
            f"{name}: verified={row.verified} "
            f"cost {row.cost_before:.2f}->{row.cost_after:.2f} "
            f"{row.spelled_us:.1f}us -> {row.scheduled_us:.1f}us "
            f"({row.speedup}x), {tri} [{device}]"
        )
    return rows


if __name__ == "__main__":
    main()
