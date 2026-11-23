"""Lower a ``lib_kernel_ir.Term`` to backend source — the Python half of T-094.

This is the ``Emitter<KernelTerm>`` analogue from plan PL-0004 Phase 3,
kept deliberately thin: the *search* stays lowering-agnostic
(``search.py``'s gate → bound → measure never sees a backend), and
lowering is an opt-in post-search step.  ``compile-triton`` and
``compile-cuda`` already exist as stdin-Term-JSON → stdout-source
binaries in the sibling ``engine-compile-*`` checkouts; this module is
the subprocess bridge plus the manifest emission, mirroring the
``source-onnx`` resolution pattern in mlopt-product's
``adapters/onnx_rust.py``.

Honest bounds, stated up front:

* **Named refusals propagate verbatim.**  A term the emitter cannot
  lower exits nonzero with its reason on stderr (``not a lowerable
  term: …``); that text is raised unchanged inside :class:`LowerError`.
  There is no silent fallback to eager — a decline is the answer.
* **Verify stays at the Term-denotation level.**  ``KernelSink.verify``
  is the executor-equivalence gate (in-process torch, rel-diff); the
  emitted Triton/CUDA source has no runner in this repo, so there is no
  artifact-level verify yet.  Measured wall-time remains the search
  referee *before* lowering; ``verify_lowered`` re-denotes the term —
  it cannot execute the emitted source.  See the module docstring of
  ``verify_lowered`` for the split.
* **The manifest is a mirror, not an import.**  ``mlopt.manifest`` does
  not exist yet (parallel work in mlopt-product); the dicts emitted
  here follow the ``engine-codegen`` wire schema (``Change`` Put /
  ``Manifest`` entry) by construction — ``{"op": "put", "artifact",
  "content_b64", "hash"}`` for changes and ``{"artifact", "hash"}``
  for manifest entries — so a future Rust ``codegen apply --manifest``
  consumes them unchanged.  ``hash`` is BLAKE3-256 hex, matching
  ``engine-codegen``'s ``ContentHash``.
* **CUDA host shim is inline.**  ``compile-cuda`` emits the
  ``cuda_emit_host`` helper namespace *inside* the same translation
  unit (``engine-compile-cuda/src/codegen.rs`` / ``contract.rs``), so
  the ``.cu`` artifact is self-contained — no companion host artifact
  is needed.  Documented here so the gap is a decision, not an
  omission.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal

import blake3
from lib_kernel_ir import Term, to_json

__all__ = [
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "LowerError",
    "lower_candidate",
    "lower_term",
    "verify_lowered",
]

Target = Literal["triton", "cuda"]

#: Provenance stamped into every manifest this module emits.
GENERATOR_NAME = "mlopt-kernel-lower"
GENERATOR_VERSION = "0.1.0dev"

#: kernelopt repo root — ``src/mlopt_kernel/lower.py`` → parents[2].
_KERNELOPT_ROOT = Path(__file__).resolve().parents[2]

_LOWER_TIMEOUT_S = 900

#: Per-target emitter spec: binary name, artifact extension, and the
#: cargo package that builds it in the sibling checkout.
_TARGETS: dict[str, dict[str, str]] = {
    "triton": {
        "bin": "compile-triton",
        "ext": "py",
        "repo": "engine-compile-triton",
        "env": "COMPILE_TRITON_BIN",
    },
    "cuda": {
        "bin": "compile-cuda",
        "ext": "cu",
        "repo": "engine-compile-cuda",
        "env": "COMPILE_CUDA_BIN",
    },
}


class LowerError(RuntimeError):
    """A lowering attempt failed — the emitter's refusal, verbatim.

    Raised for: an unknown target, a missing/unbuildable emitter
    binary, a subprocess failure, or the emitter's own named refusal
    (nonzero exit, stderr carried unchanged).  Never a silent fallback.
    """


def _blake3_hex(data: bytes) -> str:
    """BLAKE3-256 hex — ``engine-codegen``'s ``ContentHash`` format."""
    return blake3.blake3(data).hexdigest()


def _target_spec(target: str) -> dict[str, str]:
    try:
        return _TARGETS[target]
    except KeyError:
        raise LowerError(
            f"unknown lower target {target!r} — expected one of "
            f"{sorted(_TARGETS)}"
        ) from None


def _emitter_repo(target: str) -> Path | None:
    """Return the sibling ``engine-compile-*`` checkout, if present."""
    spec = _target_spec(target)
    sibling = _KERNELOPT_ROOT.parent / spec["repo"]
    return sibling if sibling.is_dir() else None


def _bin_candidates(target: str, bin_dir: Path | None) -> list[Path]:
    """Candidate binary paths, most-explicit first."""
    spec = _target_spec(target)
    name = spec["bin"]
    cands: list[Path] = []
    if bin_dir is not None:
        cands.append(bin_dir / name)
    shared = os.environ.get("KERNEL_LOWER_BIN")
    if shared:
        cands.append(Path(shared) / name)
    repo = _emitter_repo(target)
    if repo is not None:
        cands += [
            repo / "target" / "release" / name,
            repo / "target" / "debug" / name,
        ]
    shared_target = Path.home() / ".cargo" / "shared-target"
    cands += [
        shared_target / "release" / name,
        shared_target / "debug" / name,
    ]
    return cands


def _find_binary(target: str, bin_dir: Path | None) -> Path | None:
    """Resolve the emitter binary without building.

    An explicit per-target env var (``COMPILE_TRITON_BIN`` /
    ``COMPILE_CUDA_BIN``) that does not name an executable resolves to
    ``None``; :func:`_ensure_binary` turns that into a named error
    rather than silently probing other locations.
    """
    spec = _target_spec(target)
    env = os.environ.get(spec["env"])
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    for cand in _bin_candidates(target, bin_dir):
        if cand.is_file():
            return cand
    return None


def _ensure_binary(target: str, bin_dir: Path | None) -> Path:
    """Return the emitter binary, building it when a checkout exists.

    Mirrors ``onnx_rust._ensure_binary``: env/``bin_dir``/checkout
    probes first, then ``cargo build`` in the sibling repo, then a
    named :class:`LowerError` with the build instructions.
    """
    b = _find_binary(target, bin_dir)
    if b is not None:
        return b
    spec = _target_spec(target)
    env = os.environ.get(spec["env"])
    if env:
        raise LowerError(
            f"kernel-lower: {spec['env']}={env!r} does not name an "
            "executable file"
        )
    repo = _emitter_repo(target)
    if repo is not None and shutil.which("cargo") is not None:
        proc = subprocess.run(
            ["cargo", "build", "--bin", spec["bin"]],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=_LOWER_TIMEOUT_S,
        )
        if proc.returncode == 0:
            b = _find_binary(target, bin_dir)
            if b is not None:
                return b
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise LowerError(
            f"kernel-lower: cargo build --bin {spec['bin']} failed in "
            f"{repo} (rc={proc.returncode}): "
            f"{tail[-1] if tail else 'no output'}"
        )
    raise LowerError(
        f"kernel-lower: the {spec['bin']} binary was not found — set "
        f"{spec['env']} or KERNEL_LOWER_BIN, pass bin_dir=, or clone "
        f"handl-lang/{spec['repo']} next to the kernelopt checkout "
        f"and build with `cargo build --bin {spec['bin']}`"
    )


def _slug(term_json: str, target: str) -> str:
    """Deterministic artifact slug from the term bytes + target."""
    digest = _blake3_hex(term_json.encode())[:12]
    return f"kernel_{target}_{digest}"


def _emit(term: Term, target: str, binary: Path) -> str:
    """Run the emitter: Term-JSON on stdin, source on stdout."""
    raw = to_json(term)
    cmd = [str(binary)]
    # compile-triton accepts --block-n/--block-k (defaults 64/128 —
    # what the emitter itself uses); compile-cuda takes no flags.  v1
    # passes nothing: tuning belongs to a later flag surface.
    try:
        proc = subprocess.run(
            cmd,
            input=raw,
            capture_output=True,
            text=True,
            timeout=_LOWER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise LowerError(
            f"kernel-lower: {target} emitter timed out "
            f"(>{_LOWER_TIMEOUT_S}s)"
        ) from e
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "").strip()
        raise LowerError(
            f"kernel-lower: {target} emitter refused the term: {reason}"
        )
    return proc.stdout


def lower_term(
    term: Term,
    target: Target,
    *,
    out_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Lower ``term`` to ``target`` source; return ``Change`` dicts.

    ``target`` is ``"triton"`` (a ``.py`` Triton module) or ``"cuda"``
    (a ``.cu`` translation unit — the ``cuda_emit_host`` shim is inline,
    see the module docstring).  Returns a list of one ``put`` Change
    dict::

        {"op": "put",
         "artifact": "kernel_triton_<hash12>.py",
         "content_b64": "<base64 source>",
         "hash": "<blake3-256 hex of the source bytes>"}

    The schema mirrors ``engine-codegen``'s ``ChangeRepr``/``Manifest``
    wire form (``manifest.py`` in mlopt-product is being built in
    parallel against the same schema — deliberately *not* imported
    here; the coupling is the JSON shape, not the package).

    ``out_dir``, when given, additionally writes each artifact to
    ``out_dir / artifact`` — the manifest stays the source of truth;
    the files are a convenience for inspection.

    Raises :class:`LowerError` on any refusal — never falls back.
    """
    spec = _target_spec(target)
    binary = _ensure_binary(
        target, Path(bin_dir).expanduser() if bin_dir else None
    )
    source = _emit(term, target, binary)
    term_json = to_json(term)
    slug = _slug(term_json, target)
    artifact = f"{slug}.{spec['ext']}"
    content = source.encode()
    change = {
        "op": "put",
        "artifact": artifact,
        "content_b64": base64.b64encode(content).decode(),
        "hash": _blake3_hex(content),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / artifact).write_bytes(content)
    return [change]


def lower_candidate(
    candidate: Any,
    target: Target,
    *,
    out_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Lower a searched :class:`~mlopt_kernel.search.Candidate`'s term.

    Returns a manifest dict in the ``engine-codegen`` ``Manifest``
    shape::

        {"target": "triton",
         "generator": "mlopt-kernel-lower",
         "generator_version": "0.1.0dev",
         "run_id": "<uuid>",
         "source_hashes": {"term": "<blake3 hex of term JSON>"},
         "changes": [<put Change dicts>],
         "artifacts": [{"artifact": ..., "hash": ...}],
         "verify": {"level": "term-denotation", ...}}

    ``candidate.term`` must be a :class:`lib_kernel_ir.Term` (the real
    IR, not a search stub).  Lowering is opt-in and post-search:
    ``search()`` itself is untouched.
    """
    term = getattr(candidate, "term", candidate)
    if not isinstance(term, Term):
        raise LowerError(
            "lower_candidate needs a lib_kernel_ir.Term — got "
            f"{type(term).__name__}; search Candidates built on stub "
            "terms cannot be lowered"
        )
    changes = lower_term(term, target, out_dir=out_dir, bin_dir=bin_dir)
    term_json = to_json(term)
    return {
        "target": target,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "run_id": run_id or uuid.uuid4().hex,
        "source_hashes": {"term": _blake3_hex(term_json.encode())},
        "changes": changes,
        "artifacts": [
            {"artifact": c["artifact"], "hash": c["hash"]}
            for c in changes
        ],
        "verify": verify_manifest_block(),
    }


def verify_manifest_block() -> dict[str, Any]:
    """Return the verify-semantics record stamped into every manifest.

    The split (plan PL-0004 kernelopt bullet):

    * ``KernelSink.verify`` — executor equivalence, in-process torch,
      rel-diff gate.  That is the *search-time* gate: it compares the
      candidate's denotation against the incumbent before a term ever
      reaches an emitter.
    * ``Emitter::verify`` analogue — would run the *emitted source*
      (Triton/CUDA) against the same probes.  No Triton/CUDA runner
      exists in this repo yet, so artifact-level verify is deferred.
    * v1 verify therefore stays at Term-denotation level:
      ``verify_lowered`` re-denotes the term under an env.  Measured
      wall-time remains the referee, and it happens in ``search()``
      before lowering — never after.
    """
    return {
        "level": "term-denotation",
        "executor_gate": "KernelSink.verify (in-process torch rel-diff)",
        "artifact_gate": None,
        "artifact_gate_reason": (
            "no Triton/CUDA runner exists yet — emitted source is "
            "unverified beyond the term's denotation"
        ),
        "referee": "measured wall-time in search(), pre-materialize",
    }


def verify_lowered(term: Term, env: dict[str, Any]) -> Any:
    """Re-denote ``term`` under ``env`` — the v1 verify ceiling.

    This is *not* an artifact verify: it cannot execute the emitted
    Triton/CUDA source (no runner exists).  What it does prove is that
    the term being lowered still denotes — the same denotation the
    search gate checked.  Cheap (torch ops only), so callers that want
    a post-lowering sanity check can afford it; the returned tensor is
    the term's value under ``env``, suitable for a cosine/rel-diff
    comparison against an incumbent computed the same way.

    For the real artifact-level verify, a Triton/CUDA runner is the
    missing piece — tracked as a later task alongside the Rust
    ``Emitter<KernelTerm>`` wiring.
    """
    if not isinstance(term, Term):
        raise LowerError(
            f"verify_lowered needs a lib_kernel_ir.Term, got "
            f"{type(term).__name__}"
        )
    return term.denote(env)
