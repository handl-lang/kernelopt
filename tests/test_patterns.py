"""Whole-subgraph lowering patterns — residual projection.

The ``mlopt_kernel.patterns`` spike (plan 0039 phase 3): structural
subgraph detection at lowering time — ``add(x, linear/matmul(y, …))``
lowers to one compiled ``x + proj(y)`` region instead of the spelled
GEMM-then-add dispatch.  Verify-first through ``KernelSink.verify``:
the ``compile_fused=False`` executor is the spelled reference, the
default executor the fused candidate — same IR, different schedule.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_kernel import KernelSink
from mlopt_kernel.patterns import (
    ResidualProjRunner,
    match_residual_proj,
)

X = Var("x", TensorType((4, 8)))
Y = Var("y", TensorType((4, 8)))
W = Param("w", TensorType((8, 8)))
B = Param("b", TensorType((8,)))

#: Param name → Param leaf, for building terms and feeding values.
PARAMS = {"w": W, "b": B}


def _ir(term, inputs):
    return IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
    )


def _verify(term, inputs, param_names=()):
    """Spelled (eager) vs pattern-fused executors on one IR.

    Returns ``(report, opt_executor)`` — fp64 feeds, randn params.
    """
    pvals = {
        n: torch.randn(tuple(PARAMS[n].typ.shape), dtype=torch.float64)
        for n in param_names
    }
    ir = _ir(term, inputs)
    sink = KernelSink()
    ref = KernelSink(compile_fused=False).lower(ir, pvals)
    opt = sink.lower(ir, pvals)
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    return sink.verify(ref, opt, feeds), opt


def _res_linear_term(with_bias=True):
    proj = (
        Op.make("linear", Y, W, B)
        if with_bias
        else Op.make("linear", Y, W)
    )
    return Op.make("add", X, proj)


class TestMatch:
    def test_linear_with_bias(self):
        plan = match_residual_proj(_res_linear_term())
        assert plan is not None
        assert plan["kind"] == "linear"
        assert plan["residual"] == X
        assert len(plan["proj_args"]) == 3

    def test_linear_without_bias(self):
        plan = match_residual_proj(_res_linear_term(with_bias=False))
        assert plan is not None
        assert plan["kind"] == "linear"
        assert len(plan["proj_args"]) == 2

    def test_matmul_spelling(self):
        t = Op.make("add", X, Op.make("matmul", Y, W))
        plan = match_residual_proj(t)
        assert plan is not None
        assert plan["kind"] == "matmul"

    def test_swapped_operand_order(self):
        t = Op.make("add", Op.make("linear", Y, W, B), X)
        plan = match_residual_proj(t)
        assert plan is not None
        assert plan["residual"] == X

    def test_nested_sites(self):
        """``x + lin(y + lin(y))`` — both adds are sites."""
        inner = Op.make("add", Y, Op.make("linear", Y, W, B))
        t = Op.make("add", X, Op.make("linear", inner, W, B))
        plan = match_residual_proj(t)
        assert plan is not None
        inner_plan = match_residual_proj(t.args[1].args[0])
        assert inner_plan is not None

    def test_declines_plain_add(self):
        assert match_residual_proj(Op.make("add", X, Y)) is None

    def test_declines_non_add_head(self):
        t = Op.make("mul", X, Op.make("linear", Y, W, B))
        assert match_residual_proj(t) is None
        assert match_residual_proj(X) is None
        assert match_residual_proj("not a term") is None

    def test_declines_attributed_add(self):
        t = Op.make("add", X, Op.make("linear", Y, W, B), alpha=1.0)
        assert match_residual_proj(t) is None

    def test_declines_attributed_or_arity_proj(self):
        t = Op.make("add", X, Op.make("linear", Y, W, B, extra=1))
        assert match_residual_proj(t) is None
        t2 = Op.make("add", X, Op.make("linear", Y, W, B, X))
        assert match_residual_proj(t2) is None
        t3 = Op.make("add", X, Op.make("matmul", Y, W, B))
        assert match_residual_proj(t3) is None

    def test_declines_other_proj_like_ops(self):
        for op in ("conv2d", "sdpa", "softmax", "layer_norm"):
            t = Op.make("add", X, Op.make(op, Y, W))
            assert match_residual_proj(t) is None, op


class TestVerify:
    """Verify-first: spelled reference vs fused candidate, fp64."""

    def test_residual_linear_bias(self):
        rep, opt = _verify(_res_linear_term(), [X, Y], ("w", "b"))
        assert rep.passed and rep.max_abs < 1e-9
        assert any(v is not None for v in opt._patterns.values())

    def test_residual_linear_no_bias(self):
        rep, _o = _verify(
            _res_linear_term(with_bias=False), [X, Y], ("w",)
        )
        assert rep.passed and rep.max_abs < 1e-9

    def test_residual_matmul(self):
        rep, _o = _verify(
            Op.make("add", X, Op.make("matmul", Y, W)),
            [X, Y],
            ("w",),
        )
        assert rep.passed and rep.max_abs < 1e-9

    def test_swapped_order(self):
        rep, _o = _verify(
            Op.make("add", Op.make("linear", Y, W, B), X),
            [X, Y],
            ("w", "b"),
        )
        assert rep.passed and rep.max_abs < 1e-9

    def test_nested_sites_both_fuse(self):
        inner = Op.make("add", Y, Op.make("linear", Y, W, B))
        t = Op.make("add", X, Op.make("linear", inner, W, B))
        rep, opt = _verify(t, [X, Y], ("w", "b"))
        assert rep.passed and rep.max_abs < 1e-9
        fused = [v for v in opt._patterns.values() if v is not None]
        assert len(fused) == 2

    def test_shared_subterm(self):
        """``x + lin(x)`` — the residual IS the projection input."""
        t = Op.make("add", X, Op.make("linear", X, W, B))
        rep, _o = _verify(t, [X], ("w", "b"))
        assert rep.passed and rep.max_abs < 1e-9

    def test_eager_path_ignores_patterns(self):
        """``compile_fused=False`` never builds a pattern runner."""
        ir = _ir(_res_linear_term(), [X, Y])
        ex = KernelSink(compile_fused=False).lower(
            ir, {"w": torch.randn(8, 8), "b": torch.randn(8)}
        )
        assert ex._patterns == {}

    def test_decline_site_still_runs(self):
        """A non-matching ``add`` dispatches spelled under both arms."""
        t = Op.make("add", X, Op.make("mul", Y, Y))
        rep, opt = _verify(t, [X, Y])
        assert rep.passed
        assert all(v is None for v in opt._patterns.values())


class TestRunner:
    def test_runner_kind_dispatch(self):
        """The plan's kind picks the compiled template."""
        plan = match_residual_proj(_res_linear_term())
        assert ResidualProjRunner(plan)._plan["kind"] == "linear"
        plan2 = match_residual_proj(
            Op.make("add", X, Op.make("matmul", Y, W))
        )
        assert ResidualProjRunner(plan2)._plan["kind"] == "matmul"

    def test_real_lifted_term(self):
        """A ``torch.export``-lifted residual MLP detects + verifies.

        The spelling ``TorchSource`` actually emits for a transformer
        sub-layer — atomic ``linear`` + residual ``add`` — is the
        pattern's home corpus (the r4 whisper/t5 decoder census:
        ``linear`` x10 closing every sub-layer with an ``add``).
        """
        import torch.nn as nn
        from kit_torch.adapters import TorchSource

        class Block(nn.Module):
            def __init__(self, d=32):
                super().__init__()
                self.fc1 = nn.Linear(d, 4 * d)
                self.fc2 = nn.Linear(4 * d, d)
                self.proj = nn.Linear(d, d)

            def forward(self, x):
                h = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
                return x + self.proj(h) + self.fc2(self.fc1(x))

        ir, leaves = TorchSource().to_ir(
            Block().eval(), (torch.randn(2, 4, 32),)
        )
        sites = []

        def walk(t):
            if isinstance(t, Op):
                if match_residual_proj(t) is not None:
                    sites.append(t)
                for a in t.args:
                    walk(a)

        walk(ir.root)
        assert len(sites) >= 2  # x + proj(h) and the outer add

        leaves = {k: v.double() for k, v in leaves.items()}
        sink = KernelSink()
        inputs = list(ir.inputs)
        ref = KernelSink(compile_fused=False).lower(ir, leaves)
        opt = sink.lower(ir, leaves)
        feeds = tuple(
            torch.randn(tuple(v.typ.shape), dtype=torch.float64)
            for v in inputs
        )
        rep = sink.verify(ref, opt, feeds)
        assert rep.passed
        assert (
            sum(1 for v in opt._patterns.values() if v is not None) >= 2
        )

    def test_runner_evaluates_through_channel(self):
        """``run(ev)`` resolves every leaf through the operand channel."""
        plan = match_residual_proj(
            Op.make("add", X, Op.make("matmul", Y, W))
        )
        r = ResidualProjRunner(plan)
        vals = {
            X: torch.ones(4, 8),
            Y: torch.ones(4, 8),
            W: torch.eye(8),
        }
        out = r.run(lambda t: vals[t])
        expected = vals[X] + vals[Y] @ vals[W]
        assert torch.allclose(out, expected)
