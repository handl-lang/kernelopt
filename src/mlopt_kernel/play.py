# ruff: noqa: RUF002 — math notation in docstrings.
"""``mlopt_kernel.play`` — the ``kernel`` domain, explicitly seated.

The kernelopt board (plan 0032 phases 3–4): candidates are
codegen-law applications — named fused-kernel ops and strength
reductions from :data:`mlopt_kernel.laws.KERNEL_RULES` — the referee
is the gauntlet (``kit_discovery.verifier.verify_law`` derives the
scheduled candidate against the spelled source), and the score is
:func:`mlopt_kernel.cost.kernel_cost` improvement.  Cases are spelled
programs carrying live fusion/strength-reduction sites plus honest
flat boards; ``deliver`` lowers the winning extraction through
:class:`mlopt_kernel.KernelSink` and verifies it numerically —
verify-first.  The design note is
``notes/kernel-engineer-board.md`` in this repo.

This module is the *plugin*: :func:`register` seats the domain into
:mod:`kit_discovery.play`'s ``DOMAINS`` (through the public
:func:`kit_discovery.play.spec_domain` seam) and the kernel
board's players into the gamekit registry (through
:func:`gamekit.players.registry.register_player`).  Registration is
explicit — importing this module mutates nothing; ``main`` calls
``register()`` itself, and programmatic consumers (the prior
trainer, the transfer driver, tests) call it at their entry point.
Run the game::

    python -m mlopt_kernel.play --domain kernel \
        --train-cases 40 --eval-cases 10 --budget 8
    python -m mlopt_kernel.play --domain kernel --deliver

Every ``kit_discovery.play`` flag works — ``--player <name>``
resolves the registered seats (``linear``/``linear-greedy``/
``linear-cold``/``random``/``playbook`` all play ``kernel``).
"""

from __future__ import annotations

import random
from typing import Any

from discoverykit import lawdata
from gamekit.gamedata import bucket as _bucket
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    RelFeats,
    StateBroadcast,
    register_legal,
    register_part,
)
from gamekit.players.registry import PlayerSpec, register_player
from kit_discovery.play import DOMAINS, spec_domain
from mlopt_core.ir import Const, Op, TensorType, Var

from mlopt_kernel.weights import KERNEL_PLAYER_WEIGHTS

__all__ = [
    "KERNEL_FEATURES",
    "KERNEL_SPEC",
    "main",
    "register",
]


def _kernel_cases(seed: int, n: int) -> list:
    """Generate spelled programs over the kernel bank's live sites.

    Each archetype exercises a different paying move — the gate
    fusion (``a·sigmoid(b)``), the softsign fusion (``a/(|b|+c)``), the
    FMA fusion (``a·b + c``), and the strength reductions — plus
    compositions and honest flat boards, so the player's signal is
    *which* move pays, not whether any does.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 8)))
    y = Var("y", TensorType((4, 8)))
    w = Var("w", TensorType((8, 4)))
    v = Var("v", TensorType((4, 4)))
    b = Var("b", TensorType((4,)))
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    qk = Op.make("matmul", x, w)
    pool = [
        ("silu", Op.make("mul", x, Op.make("sigmoid", x))),
        (
            "gate",
            Op.make("mul", y, Op.make("sigmoid", x)),
        ),
        (
            "softsign",
            Op.make(
                "div",
                x,
                Op.make("add", Op.make("abs", x), Const(1)),
            ),
        ),
        (
            "fma",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
        ),
        (
            "swiglu",
            Op.make("mul", Op.make("mul", x, Op.make("sigmoid", x)), y),
        ),
        (
            "softmax",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=-1, keepdim=True),
            ),
        ),
        (
            "layernorm",
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
        ),
        (
            "mm_bias",
            Op.make("add", Op.make("matmul", x, w), b),
        ),
        (
            "mm_sb",
            Op.make(
                "add",
                Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                b,
            ),
        ),
        (
            "mm_scale",
            Op.make("mul", qk, Const(0.125)),
        ),
        (
            "mm_gelu",
            Op.make("gelu", qk),
        ),
        (
            "rmsnorm",
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
        ),
        (
            "softmax_mm",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make("exp", qk),
                    Op.make(
                        "sum",
                        Op.make("exp", qk),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                v,
            ),
        ),
        (
            "sdpa_chain",
            Op.make(
                "matmul",
                Op.make(
                    "div",
                    Op.make("exp", Op.make("mul", qk, Const(0.5))),
                    Op.make(
                        "sum",
                        Op.make("exp", Op.make("mul", qk, Const(0.5))),
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                v,
            ),
        ),
        (
            "fma_tanh",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
        ),
        (
            "double_gate",
            Op.make(
                "mul",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("sigmoid", y),
            ),
        ),
        ("square", Op.make("pow", x, Const(2))),
        ("recip", Op.make("div", x, Const(4.0))),
        (
            "mixed",
            Op.make(
                "add",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("div", Op.make("pow", x, Const(2)), Const(4.0)),
            ),
        ),
        (
            "flat",
            Op.make("matmul", x, w),
        ),
    ]
    out = []
    for i in range(n):
        kind, term = pool[rng.randrange(len(pool))]
        names = {v.name for v in _vars_of(term)}
        vars_in = [var for var in (x, y, w, v, b) if var.name in names]
        out.append(
            (
                f"kernel{i}:{kind}",
                term,
                kind,
                (),
                (),
                {"inputs": vars_in},
            )
        )
    return out


def _vars_of(term: Any):
    """Yield every ``Var`` leaf reachable in *term*."""
    if isinstance(term, Var):
        yield term
    elif isinstance(term, Op):
        for a in term.args:
            yield from _vars_of(a)


def _kernel_board(case: Any) -> Any:
    """Board a kernel case under the codegen bank + gauntlet referee.

    ``step_cost=0.05`` is the kernel board's oracle-call economics:
    firing a codegen law is a compile/measure call, so moves pay a
    small charge and ``stop`` — declining the remaining evaluations —
    is the skill the board scores.
    """
    from discoverykit.verifier import verify_law

    from mlopt_kernel import KernelBoard

    return KernelBoard(case[1], verifier=verify_law, step_cost=0.05)


def _kernel_legal(state: Any) -> tuple:
    """Return the codegen laws plus ``stop`` — ending is a move."""
    return (*state.actions, "stop")


#: The op vocabulary the kernel cases draw from — the board's
#: site-detection signal, broadcast as binary ``st:op:*`` flags.
_KERNEL_OP_FEATURES: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "sigmoid",
    "tanh",
    "pow",
    "matmul",
    "exp",
    "relu",
    "mean",
    "sum",
    "rsqrt",
    "gelu",
    "erf",
)


def _kernel_state(state: Any) -> dict[str, float]:
    """Return the ``st:*`` broadcast — progress, improvement, sites."""
    out = {
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
        "st:const": float(state.has_const),
        "st:sites": float(len(state.matches)),
    }
    for op in _KERNEL_OP_FEATURES:
        out[f"st:op:{op}"] = float(op in state.ops)
    return out


def _kernel_rule_bucket(name: Any) -> int:
    """Bucket a rule name into ``META_ARENA_HASH_BUCKETS["rule"]``.

    The canonical :func:`gamekit.gamedata.bucket` bound at import —
    the shipped ``KERNEL_PLAYER_WEIGHTS`` was trained against this
    hash, so the rule-name → ``a:r:*`` mapping must not drift.
    """
    return _bucket(name, "rule", lawdata.META_ARENA_HASH_BUCKETS)


def _kernel_act(
    action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the fired law's identity bucket — action features."""
    return {f"a:r:{_kernel_rule_bucket(action):02d}": 1.0}


def _kernel_rel(
    state: Any, action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the state-x-action interaction — does this law apply here?.

    ``x:m`` flags a rule whose checked pattern occurs in the program
    (the board's ``matches`` — a matching codegen is the one that can
    pay its compile cost); ``x:stop`` gives the stop move its own
    weight, and ``x:stop:imp`` prices stopping against the
    improvement already banked.
    """
    if action == "stop":
        return {
            "x:stop": 1.0,
            "x:stop:imp": _row.get("st:improve", 0.0),
        }
    return {"x:m": float(action in state.matches)}


KERNEL_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    "st:const",
    "st:sites",
    *(f"st:op:{op}" for op in _KERNEL_OP_FEATURES),
    "x:m",
    "x:stop",
    "x:stop:imp",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)

#: The kernel spec — intercept, progress/improvement/site broadcast,
#: the match interaction, and the rule bucket; the shipped prior is
#: ``mlopt_kernel.weights.KERNEL_PLAYER_WEIGHTS`` (match-oracle
#: imitation + REINFORCE; learned-greedy 0.646 > scripted 0.276 on
#: held-out).
KERNEL_SPEC = DomainSpec.build(
    features=KERNEL_FEATURES,
    legal=_kernel_legal,
    parts=(
        Bias(),
        StateBroadcast(_kernel_state),
        RelFeats(_kernel_rel),
        ActionFeats(_kernel_act),
    ),
    weights=KERNEL_PLAYER_WEIGHTS,
)


def _scripted_kernel() -> Any:
    """Play the playbook: fire every matching law once, bank order.

    ``kern_fuse_*`` lhs heads name the op the pattern needs; a law
    whose head is absent can never match, so firing it wastes the
    step — matching-head-first is the honest scripted line.
    """
    from mlopt_kernel import KERNEL_RULES

    heads = {r.name: getattr(r.lhs, "op", None) for r in KERNEL_RULES}
    played: set = set()

    def go(state: Any) -> Any | None:
        if state.done:
            return None
        for name in state.actions:
            if name not in played and heads.get(name) in state.ops:
                played.add(name)
                return name
        for name in state.actions:
            if name not in played:
                played.add(name)
                return name
        return None

    return go


def _kernel_deliver(board: Any, case: Any, traj: Any) -> dict:
    """Lower the winning extraction through ``KernelSink``; verify it.

    The deliver half of the kernel board: the spelled source and the
    candidate the episode's extraction picked are both lowered in the
    sink's runtime and compared on random feeds — verify-first.

    Honesty contract: a measurement arm, not the enforced
    ``Optimizer.lower(verify=True)`` shipping gate — it reports the
    verdict rather than raising on it.  ``delivered`` is ``rep.passed``:
    a lowered-but-unverified candidate is marked ``delivered_unverified``
    and is NEVER reported as ``delivered``.
    """
    import torch
    from mlopt_core.ir import IR

    from mlopt_kernel import KernelSink

    best = board.extract_best()
    if best is None:
        return {"delivered": False, "reward": traj.total}
    inputs = list(case[5]["inputs"])
    sink = KernelSink()
    ref = sink.lower(
        IR(root=case[1], inputs=inputs, params={}), params={}
    )
    opt = sink.lower(IR(root=best, inputs=inputs, params={}), params={})
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    rep = sink.verify(ref, opt, feeds)
    t = traj.terminal
    out = {
        "delivered": bool(rep.passed),
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }
    if not rep.passed:
        out["delivered_unverified"] = True
    return out


# ---------------------------------------------------------------------------
#  Registration — the phase-4 plugin seam, made explicit
# ---------------------------------------------------------------------------
#
#  ``register()`` is the only place the domain seats itself: the
#  feature/legal parts, ``play.DOMAINS["kernel"]``, and the kernel
#  board's players into the gamekit registry.  ``linear*``/``random``
#  need no ``make`` — their shipped factories are spec-driven
#  (``linear`` reads ``DOMAINS["kernel"].spec``; ``random`` is
#  uniform); ``playbook`` supplies its own because the scripted line
#  lives where the domain lives.  Idempotent — the gamekit seams
#  overwrite by name — so every entry point may call it.


def _playbook_make(
    domain: str, *, spec: PlayerSpec, seed: int = 0, ctx: Any = None
) -> Any:
    """Return the kernel playbook's arm factory — the scripted line."""
    del spec, seed, ctx
    assert domain == "kernel"
    return lambda i: _scripted_kernel()


def register() -> None:
    """Seat the kernel domain and its players — call, don't import."""
    register_part("kernel:state", lambda: StateBroadcast(_kernel_state))
    register_part("kernel:action", lambda: ActionFeats(_kernel_act))
    register_part("kernel:rel", lambda: RelFeats(_kernel_rel))
    register_legal("kernel", _kernel_legal)

    DOMAINS["kernel"] = spec_domain(
        _kernel_cases,
        _kernel_board,
        KERNEL_SPEC,
        scripted=_scripted_kernel,
        deliver=_kernel_deliver,
    )

    register_player(
        PlayerSpec(
            name="linear",
            kind="linear",
            boards=("kernel",),
            schema="the kernel DomainSpec row — KERNEL_SPEC",
            train="scripts/train_kernel_prior.py",
            artifact="mlopt_kernel.weights.KERNEL_PLAYER_WEIGHTS",
            record="kernel 0.646 held-out (greedy) — beats scripted 0.276",
        )
    )
    register_player(
        PlayerSpec(
            name="linear-greedy",
            kind="linear",
            boards=("kernel",),
            schema="same as ``linear`` — argmax decode",
            train="scripts/train_kernel_prior.py",
            artifact="mlopt_kernel.weights.KERNEL_PLAYER_WEIGHTS",
            record="kernel 0.646 held-out — the shipped v2 prior",
            variant={"greedy": True},
        )
    )
    register_player(
        PlayerSpec(
            name="linear-cold",
            kind="linear",
            boards=("kernel",),
            schema="same as ``linear`` — ``weights={}``",
            train="— (the untrained control)",
            artifact="— (the point is no weights)",
            record="kernel cold start — the control the prior must beat",
            variant={"cold": True},
        )
    )
    register_player(
        PlayerSpec(
            name="random",
            kind="chance",
            boards=("kernel",),
            schema="— (no features; the move is a uniform draw)",
            train="— (nothing to train)",
            artifact="— (no weights)",
            record="kernel -0.045 held-out reward/ep — the floor",
        )
    )
    register_player(
        PlayerSpec(
            name="playbook",
            kind="scripted",
            boards=("kernel",),
            schema="— (a fixed line, not a scorer)",
            train="— (hand-written; nothing to train)",
            artifact="— (no weights)",
            record="kernel 0.276 held-out reward/ep — matching-head-first",
        ),
        make=_playbook_make,
    )


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``kit_discovery.play.main`` — the domain is seated."""
    register()
    from kit_discovery.play import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
