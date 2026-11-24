"""Chunked-scan probe — does the fused schedule close the FLA gap?

Follow-up to ``kernel_vocab_probe`` / ``docs/results_kernel_vocab.md``:
the retnet carrier lowers through ``KernelSink`` and verifies, but the
per-op dispatch tape + CUDA graph lands 1.011 ms vs the batched
``ChunkedDecayModule`` carrier's 0.537 ms vs FLA ~0.28 ms (carried
retro).  The named residual was *schedule*: a fused chunked scan over
the ``concat``-of-chunks — ``om_decayed`` alone only expresses
``(s⊙d)@v`` per chunk.

This probe measures the new ladder:

1. **The fold** — the extracted carrier term (vendored ``_saturate``,
   T=512/K=4 — the audit's config) gains ``om_chunk_scan`` members via
   the ``kern_fuse_chunk_scan`` family; extraction under
   ``kernel_cost`` must pick the scan at the root.
2. **Verify-first** — the scheduled executor verifies against the
   spelled fp32 module, both compiled-scheduled and eager-cat paths.
3. **The arm table** — spelled / per-op tape / tape+graph / batched
   carrier+graph / scheduled scan / scan+graph — honest rows against
   the 0.537 ms carrier arm and the ~0.28 ms FLA reference.

Run (needs the *parent* env — ``kit_carriers`` is deliberately not a
kernelopt dep)::

    PYTHONPATH=/path/to/kernelopt/src \
        /path/to/graphopt/.venv/bin/python \
        bench/drivers/probes/kernel_scan_probe.py
"""

from __future__ import annotations

import os
import sys
from typing import cast

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_vocab_probe import (
    _capture_manual,
    _ops_of,
    _saturate,
    _time_module,
)

GAMMA = 0.9


def _fold_scan(best):
    """Fire the chunk-scan family on the extracted term; re-extract.

    Returns ``(scheduled_term, fired_classes)`` — the fold is a single
    ``EGraph.apply_rule`` pass: the check parses through nested
    ``concat`` children, so every concat e-class in the spine mints
    its ``om_chunk_scan`` member in one sweep.
    """
    from mlopt_core.egraph import EGraph

    from mlopt_kernel import KERNEL_RULES, kernel_cost

    eg = EGraph()
    root = eg.add_term(best)
    n0 = eg.n_enodes
    for r in KERNEL_RULES:
        if r.name.startswith("kern_fuse_chunk_scan"):
            eg.apply_rule(r, root)
    after = eg.n_enodes - n0
    out = eg.extract_best(root, kernel_cost)
    return eg, root, out, after


def main() -> None:
    """Run the fold + verify + the retnet end-to-end arm table."""
    import copy

    from kit_torch.adapters import TorchSink
    from mlopt_core.ir import IR
    from mlopt_orchestrator.optimize import _lower_extracted

    from mlopt_kernel import KernelSink, kernel_cost

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = 30

    print("=== retnet T=512 K=4 (the audit's config) ===")
    sat = _saturate(512, 4, 14, 200, 60)
    ops = _ops_of(sat["best"])
    print(
        f"extracted: root={sat['root_op']} distinct={len(ops)} "
        f"nodes={sum(ops.values())} sat={sat['sat_secs']:.1f}s"
    )
    print(
        f"  kernel_cost(spelled extract) = {kernel_cost(sat['best']):.0f}"
    )

    _eg, _root, sched, added = _fold_scan(sat["best"])
    print(
        f"scan fold: +{added} enodes -> root={sched.op} "
        f"kernel_cost={kernel_cost(sched):.0f}"
    )
    sched_ops = _ops_of(sched)
    print(f"  scheduled ops: {sched_ops}")

    ex_s = IR(
        root=sat["best"],
        inputs=list(sat["ir"].inputs),
        input_names=set(sat["ir"].input_names),
        params=dict(sat["ir"].params),
    )
    ex_f = IR(
        root=sched,
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

    tsink = TorchSink()
    ksink = KernelSink()
    carrier32 = cast(nn.Module, tsink.lower(ex_s, params=leaves32)).to(
        device
    )
    fused32 = _lower_extracted(ex_s.root, ex_s, leaves32, tsink).to(
        device
    )
    ktape = ksink.lower(ex_s, params=leaves32)  # the per-op tape arm
    kscan = ksink.lower(ex_f, params=leaves32)  # the scheduled scan
    kscan_e = KernelSink(compile_fused=False).lower(
        ex_f, params=leaves32
    )

    # verify-first — never ship on a law's say-so.
    rep_s = ksink.verify(base32, kscan, (x32,))
    rep_se = ksink.verify(base32, kscan_e, (x32,))
    rep_t = ksink.verify(base32, ktape, (x32,))
    rep_c = tsink.verify(base32, carrier32, (x32,))
    print(
        f"verify32: scan rel {rep_s.max_rel:.2e} passed={rep_s.passed} "
        f"| scan-eager rel {rep_se.max_rel:.2e} | tape rel "
        f"{rep_t.max_rel:.2e} | carrier rel {rep_c.max_rel:.2e}"
    )

    arms: dict = {
        "spelled-eager32": lambda: base32(x32),
        "carrier(gen)32": lambda: carrier32(x32),
        "fused(exec)32": lambda: fused32(x32),
        "kernel_tape": lambda: ktape(x32),
        "kernel_scan": lambda: kscan(x32),
        "kernel_scan(eager)": lambda: kscan_e(x32),
    }
    try:
        ktape_g = _capture_manual(lambda t_: ktape(t_), x32)
        arms["kernel_tape+graph"] = lambda: ktape_g(x32)
    except Exception as e:
        print(f"  kernel_tape graph capture FAILED: {e}")
    try:
        car_g = _capture_manual(lambda t_: carrier32(t_), x32)
        arms["carrier32+graph(man)"] = lambda: car_g(x32)
    except Exception as e:
        print(f"  carrier32 graph capture FAILED: {e}")
    try:
        kscan_g = _capture_manual(lambda t_: kscan(t_), x32)
        arms["kernel_scan+graph"] = lambda: kscan_g(x32)
    except Exception as e:
        print(f"  kernel_scan graph capture FAILED: {e}")

    def _fg32():
        if not fused32.is_graph_captured:
            fused32.capture_cuda_graph(x32)
        return fused32(x32)

    if hasattr(fused32, "capture_cuda_graph"):
        arms["fused32+graph"] = _fg32

    print(f"{'arm':>24} {'med ms':>9}")
    for name, fn in arms.items():
        try:
            with torch.no_grad():
                fn()
            ms = _time_module(lambda *_, f=fn: f(), (x32,), reps) / 1e3
            print(f"{name:>24} {ms:9.3f}")
        except Exception as e:
            print(f"{name:>24}  FAILED: {type(e).__name__}: {e}")
    print(
        "\nreferences: carrier32+graph 0.537 ms (vocab probe), "
        "fused32+graph incumbent 0.454 ms (audit), "
        "fla-chunk ~0.28 ms (carried retro — fla-core absent)"
    )


if __name__ == "__main__":
    main()
