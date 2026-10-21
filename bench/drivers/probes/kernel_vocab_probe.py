"""Kernel-vocabulary probe — does the extracted retnet carrier lower?

``docs/results_kernel_check.md`` (the audit moved here with the
split) found the extracted retnet carrier could not lower through
``KernelSink`` at all: 8 of its 14 ops were outside
``supported_ops`` — ``decay``, ``decay_vec``, ``om_decayed``,
``split``, ``concat``, ``linear``, ``contiguous``, ``detach_``.
This probe re-asks the question on the widened table:

1. **Per-op verify** — every new binding evaluates its torch
   reference; ``om_decayed`` additionally verifies against its
   spelled ``matmul(mul(s,d),v)`` site (verify-first).
2. **End-to-end** — the *real* extracted carrier (the probe's own
   ``_saturate``, T=512/K=4 — the audit's config, vendored verbatim)
   is lowered through ``KernelSink`` and timed against the
   incumbent delivery path (``fused32+graph`` ≈ 0.454 ms) and the
   FLA reference (~0.28 ms, carried retro — ``fla-core`` is not
   installed, same caveat as the audit doc).

The honest-negative discipline: the lowered executor is a per-op
dispatch tape with the fused ops compiled — it is NOT the FLA-class
fused chunk-scan.  Whether a manual CUDA-graph capture or a
whole-graph ``torch.compile`` of the executor recovers the
schedule is measured, not asserted.

Run (needs the *parent* env — ``mlopt_carriers``/``mlopt_cuda`` are
deliberately not kernelopt deps)::

    PYTHONPATH=/path/to/kernelopt/src \
        /path/to/research-mlopt/.venv/bin/python \
        bench/drivers/probes/kernel_vocab_probe.py
"""

from __future__ import annotations

import time
from typing import cast

import torch
import torch.nn as nn

GAMMA = 0.9


class _ChunkedRetention(nn.Module):
    """Verbatim copy of ``mlopt_discovery.zoo._ChunkedRetention``."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gamma = 0.9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-2]
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        decay = torch.tril(
            torch.pow(
                torch.tensor(
                    self.gamma, dtype=x.dtype, device=x.device
                ),
                (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0),
            )
        )
        s = self.q(x) @ self.k(x).transpose(-1, -2) / (t**0.5)
        return (s * decay) @ self.v(x)


def _saturate(t: int, k: int, iters: int, budget: int, chunk: int):
    """``wholemodel_gap_probe._saturate``, vendored verbatim."""
    from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS, om_kway_laws
    from mlopt_core.cost import dag_cost, flops_cost
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws.tensor import (
        ASSOC_MATMUL_REV,
        DIV_ADD_FACTOR,
        FACTOR_MUL,
        FACTOR_MUL_ADD,
        MATMUL_DIV_DISTRIBUTE,
        MUL_DIV_SWAP,
        SPLIT_SCALAR_RULES,
    )
    from mlopt_torch.adapters import TorchSource

    bridges = [
        MUL_DIV_SWAP,
        MATMUL_DIV_DISTRIBUTE,
        ASSOC_MATMUL_REV,
        FACTOR_MUL,
        FACTOR_MUL_ADD,
        DIV_ADD_FACTOR,
    ]
    torch.manual_seed(0)
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + bridges
        + om_kway_laws(k)
    )
    budgets = {
        "om_decayed_chunkify_rows": chunk,
        "om_decayed_chunkify_keys": chunk,
        f"om_decayed_kway_blocks_{k}": budget,
    }
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir, leaves = TorchSource().to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    t0 = time.perf_counter()
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=200_000,
        patience=8,
        rule_budgets=budgets,
    )
    secs = time.perf_counter() - t0
    best = eg.extract_best(eg.find(root), flops_cost)
    return {
        "model": model,
        "x": x,
        "ir": ir,
        "leaves": leaves,
        "best": best,
        "cost": dag_cost(best, flops_cost),
        "root_op": best.op,
        "n_enodes": eg.n_enodes,
        "sat_secs": secs,
    }


def _time_module(model, x, reps: int) -> float:
    """Median per-call latency (us) — the shared harness."""
    args = x if isinstance(x, (tuple, list)) else (x,)
    dev = args[0].device
    for _ in range(10):
        model(*args)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _capture_manual(fn, x: torch.Tensor):
    """Capture ``fn(x)`` into a CUDA graph; return a replay callable."""
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            fn(static_x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = fn(static_x)

    def call(inp: torch.Tensor) -> torch.Tensor:
        static_x.copy_(inp, non_blocking=True)
        g.replay()
        return static_out

    return call


def _ops_of(term) -> dict:
    """Distinct-op multiset of a term DAG."""
    from collections import Counter

    from mlopt_core.ir import Op

    seen: set[int] = set()
    ops: Counter = Counter()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            ops[t.op] += 1
            stack.extend(t.args)
    return dict(ops)


def _per_op_verify() -> list[tuple[str, bool, float]]:
    """Each new binding vs its torch reference — verify-first rows."""
    from mlopt_core.ir import IR, Op, Param, TensorType, Var

    from mlopt_kernel import KernelSink

    x = Var("x", TensorType((4, 8)))
    y = Var("y", TensorType((4, 8)))
    w = Var("w", TensorType((4, 8)))
    b = Var("b", TensorType((4,)))
    g = Var("g", TensorType(()))
    s = Var("s", TensorType((4, 8)))
    d = Var("d", TensorType((4, 8)))
    v = Var("v", TensorType((8, 4)))

    torch.manual_seed(0)
    xt = torch.randn(4, 8, dtype=torch.float64)
    gt = torch.tensor(0.9, dtype=torch.float64)
    wt = torch.randn(4, 8, dtype=torch.float64)
    bt = torch.randn(4, dtype=torch.float64)
    dt = torch.rand(4, 8, dtype=torch.float64)
    vt = torch.randn(8, 4, dtype=torch.float64)
    pos8 = torch.arange(8, dtype=torch.float64)

    rows = []

    def check(name, term, feeds, want_fn, params=None):
        sink = KernelSink(compile_fused=False)
        inputs = [
            vr
            for vr in (x, y, w, b, g, s, d, v)
            if vr.name in _leaf_names(term)
        ]
        ir = IR(
            root=term,
            inputs=inputs,
            input_names={vr.name for vr in inputs},
        )
        out = sink.lower(ir, params=params)(*feeds)
        want = want_fn()
        ok = torch.allclose(out, want, atol=1e-9, rtol=1e-9)
        err = (
            float((out - want).abs().max())
            if out.shape == want.shape
            else float("nan")
        )
        rows.append((name, ok, err))

    def _leaf_names(term):
        names = set()
        stack = [term]
        while stack:
            t = stack.pop()
            if isinstance(t, (Var, Param)):
                names.add(t.name)
            elif isinstance(t, Op):
                stack.extend(t.args)
        return names

    check(
        "split",
        Op.make("split", x, sizes=(2, 2), dim=0, index=1),
        (xt,),
        lambda: torch.split(xt, [2, 2], dim=0)[1],
    )
    check(
        "concat",
        Op.make("concat", x, y, dim=0),
        (xt, xt),
        lambda: torch.cat([xt, xt], dim=0),
    )
    check(
        "linear",
        Op.make(
            "linear",
            x,
            Param("lw", TensorType((4, 8))),
            Param("lb", TensorType((4,))),
        ),
        (xt,),
        lambda: torch.nn.functional.linear(xt, wt, bt),
        params={"lw": wt, "lb": bt},
    )
    check(
        "contiguous",
        Op.make("contiguous", Op.make("transpose", x)),
        (xt,),
        lambda: xt.t().contiguous(),
    )
    check(
        "detach_",
        Op.make("detach_", x),
        (xt,),
        lambda: xt.detach(),
    )
    check(
        "decay",
        Op.make("decay", g, t=8),
        (gt,),
        lambda: torch.tril(
            gt ** (pos8[:, None] - pos8[None, :]).clamp_min(0)
        ),
    )
    check(
        "decay_vec",
        Op.make("decay_vec", g, t=8, sign=-1),
        (gt,),
        lambda: gt ** (-pos8),
    )
    check(
        "om_decayed",
        Op.make("om_decayed", s, d, v),
        (xt, dt, vt),
        lambda: (xt * dt) @ vt,
    )
    return rows


def main() -> None:
    """Run per-op verify + the retnet end-to-end arm table."""
    import copy

    from mlopt_core.ir import IR
    from mlopt_orchestrator.optimize import _lower_extracted
    from mlopt_torch.adapters import TorchSink

    from mlopt_kernel import KernelSink, kernel_cost

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = 30

    print("=== per-op verify (new vocabulary) ===")
    for name, ok, err in _per_op_verify():
        print(f"  {name:>12}: verified={ok} max_abs={err:.2e}")

    print("\n=== retnet T=512 K=4 (the audit's config) ===")
    sat = _saturate(512, 4, 14, 200, 60)
    ops = _ops_of(sat["best"])
    print(
        f"extracted: root={sat['root_op']} distinct={len(ops)} "
        f"nodes={sum(ops.values())} cost={sat['cost']:,.0f} "
        f"sat={sat['sat_secs']:.1f}s"
    )
    print("  ops:", ops)
    missing = set(ops) - KernelSink().supported_ops
    print(f"  outside supported_ops: {sorted(missing) or 'NONE'}")

    tsink = TorchSink()
    ksink = KernelSink()
    ex = IR(
        root=sat["best"],
        inputs=list(sat["ir"].inputs),
        input_names=set(sat["ir"].input_names),
        params=dict(sat["ir"].params),
    )
    leaves32 = {
        n: v.detach().float().to(device)
        for n, v in sat["leaves"].items()
    }
    x32 = sat["x"].float().to(device)
    base32 = copy.deepcopy(sat["model"]).float().to(device)

    carrier32 = cast(nn.Module, tsink.lower(ex, params=leaves32)).to(
        device
    )
    fused32 = _lower_extracted(ex.root, ex, leaves32, tsink).to(device)
    kexec = ksink.lower(ex, params=leaves32)
    kexec_eager = KernelSink(compile_fused=False).lower(
        ex, params=leaves32
    )

    # verify the kernel-sink delivery FIRST (never ship on a law's
    # say-so) — against the spelled fp32 module.
    rep_k = ksink.verify(base32, kexec, (x32,))
    rep_ke = ksink.verify(base32, kexec_eager, (x32,))
    rep_c = tsink.verify(base32, carrier32, (x32,))
    print(
        f"verify32: kernel_exec rel {rep_k.max_rel:.2e} "
        f"passed={rep_k.passed} | eager rel {rep_ke.max_rel:.2e} "
        f"passed={rep_ke.passed} | carrier rel {rep_c.max_rel:.2e}"
    )

    arms: dict = {
        "spelled-eager32": lambda: base32(x32),
        "carrier(gen)32": lambda: carrier32(x32),
        "fused(exec)32": lambda: fused32(x32),
        "kernel_exec": lambda: kexec(x32),
        "kernel_exec(eager)": lambda: kexec_eager(x32),
    }
    try:
        kexec_g = _capture_manual(lambda t_: kexec(t_), x32)
        arms["kernel_exec+graph"] = lambda: kexec_g(x32)
    except Exception as e:
        print(f"  kernel_exec graph capture FAILED: {e}")
    try:
        car_g = _capture_manual(lambda t_: carrier32(t_), x32)
        arms["carrier32+graph(man)"] = lambda: car_g(x32)
    except Exception as e:
        print(f"  carrier32 graph capture FAILED: {e}")

    def _fg32():
        if not fused32.is_graph_captured:
            fused32.capture_cuda_graph(x32)
        return fused32(x32)

    if hasattr(fused32, "capture_cuda_graph"):
        arms["fused32+graph"] = _fg32

    # the whole-graph experiment: torch.compile over the executor's
    # forward — does Inductor fuse the per-op tape into one graph?
    try:
        kexec_c = torch.compile(kexec.forward)
        arms["kernel_exec+compile"] = lambda: kexec_c(x32)
    except Exception as e:
        print(f"  kernel_exec torch.compile FAILED: {e}")

    print(f"kernel_cost(extracted) = {kernel_cost(sat['best']):.0f}")
    print(f"{'arm':>24} {'med ms':>9}")
    rows = {}
    for name, fn in arms.items():
        try:
            with torch.no_grad():
                fn()
            ms = _time_module(lambda *_, f=fn: f(), (x32,), reps) / 1e3
            rows[name] = ms
            print(f"{name:>24} {ms:9.3f}")
        except Exception as e:
            rows[name] = None
            print(f"{name:>24}  FAILED: {type(e).__name__}: {e}")
    print(
        "\nreferences: fused32+graph incumbent 0.454 ms (audit), "
        "fla-chunk ~0.28 ms (carried retro — fla-core absent)"
    )


if __name__ == "__main__":
    main()
