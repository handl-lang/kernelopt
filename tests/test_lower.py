"""``mlopt_kernel.lower`` — the T-094 Python lowering path.

Two layers of tests:

* **Stub-emitter tests** (always run): a fake ``compile-triton`` /
  ``compile-cuda`` shell script on ``bin_dir`` exercises the subprocess
  bridge, the Change-dict shape, refusal propagation, and the manifest
  — no cargo, no GPU.
* **Real-emitter tests** (skip-if-binary-missing): when a built
  ``compile-triton``/``compile-cuda`` resolves without building
  (``_find_binary`` — explicit env/dir/built checkout only, never a
  cargo build inside the test run), fixture terms go through the real
  binaries.  Mirrors the honest-skip convention: a missing binary is a
  skip, not a failure.
"""

from __future__ import annotations

import base64
import os
import stat
from types import SimpleNamespace

import pytest
import torch

lib_kernel_ir = pytest.importorskip("lib_kernel_ir")

import blake3  # noqa: E402
from lib_kernel_ir import Leaf, Term, qgemv  # noqa: E402

from mlopt_kernel import lower as lower_mod  # noqa: E402
from mlopt_kernel.lower import (  # noqa: E402
    LowerError,
    lower_candidate,
    lower_term,
    verify_lowered,
)

# ---------------------------------------------------------------------------
#  Fixtures + stub emitters
# ---------------------------------------------------------------------------


def _write_stub(bin_dir, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(
        path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


@pytest.fixture
def stub_bins(tmp_path, monkeypatch):
    """A bin_dir with stub ``compile-triton``/``compile-cuda`` binaries.

    Stubs consume stdin and emit a fixed source line; ``compile-cuda``
     is swapped per-test when a refusal is needed.
    """
    _write_stub(
        tmp_path,
        "compile-triton",
        "cat >/dev/null\necho \"print('triton')\"",
    )
    _write_stub(
        tmp_path, "compile-cuda", "cat >/dev/null\necho '// cuda'"
    )
    return tmp_path


QGEMV_TERM = qgemv(256, 2)
ELEMENTWISE_TERM = Term("neg", (Leaf("x"),))


# ---------------------------------------------------------------------------
#  Stub-emitter tests — always run
# ---------------------------------------------------------------------------


def test_lower_term_triton_change_shape(stub_bins, tmp_path) -> None:
    changes = lower_term(QGEMV_TERM, "triton", bin_dir=stub_bins)
    assert len(changes) == 1
    ch = changes[0]
    assert ch["op"] == "put"
    assert ch["artifact"].startswith("kernel_triton_")
    assert ch["artifact"].endswith(".py")
    payload = base64.b64decode(ch["content_b64"])
    assert payload == b"print('triton')\n"
    assert ch["hash"] == blake3.blake3(payload).hexdigest()


def test_lower_term_cuda_change_shape(stub_bins) -> None:
    changes = lower_term(ELEMENTWISE_TERM, "cuda", bin_dir=stub_bins)
    ch = changes[0]
    assert ch["artifact"].startswith("kernel_cuda_")
    assert ch["artifact"].endswith(".cu")
    assert base64.b64decode(ch["content_b64"]) == b"// cuda\n"


def test_lower_term_is_deterministic_slug(stub_bins) -> None:
    a = lower_term(QGEMV_TERM, "triton", bin_dir=stub_bins)
    b = lower_term(QGEMV_TERM, "triton", bin_dir=stub_bins)
    assert a[0]["artifact"] == b[0]["artifact"]
    assert a[0]["hash"] == b[0]["hash"]


def test_lower_term_out_dir_writes_file(stub_bins, tmp_path) -> None:
    out = tmp_path / "out"
    changes = lower_term(
        QGEMV_TERM, "cuda", bin_dir=stub_bins, out_dir=out
    )
    written = out / changes[0]["artifact"]
    assert written.read_bytes() == base64.b64decode(
        changes[0]["content_b64"]
    )


def test_lower_term_refusal_propagates_verbatim(tmp_path) -> None:
    _write_stub(
        tmp_path,
        "compile-triton",
        "cat >/dev/null\necho 'not a lowerable term: bogus op' >&2\nexit 3",
    )
    with pytest.raises(
        LowerError, match="not a lowerable term: bogus op"
    ):
        lower_term(ELEMENTWISE_TERM, "triton", bin_dir=tmp_path)


def test_lower_term_unknown_target() -> None:
    with pytest.raises(LowerError, match="unknown lower target"):
        lower_term(ELEMENTWISE_TERM, "opencl")  # type: ignore[arg-type]


def test_lower_term_missing_binary_names_remedies(monkeypatch) -> None:
    monkeypatch.setenv(
        "COMPILE_TRITON_BIN", "/nonexistent/compile-triton"
    )
    with pytest.raises(LowerError, match="COMPILE_TRITON_BIN"):
        lower_term(ELEMENTWISE_TERM, "triton")


def test_lower_candidate_manifest_shape(stub_bins) -> None:
    cand = SimpleNamespace(term=QGEMV_TERM, label="qgemv-2")
    manifest = lower_candidate(cand, "triton", bin_dir=stub_bins)
    assert manifest["target"] == "triton"
    assert manifest["generator"] == "mlopt-kernel-lower"
    assert manifest["run_id"]
    assert (
        manifest["source_hashes"]["term"]
        == blake3.blake3(
            lib_kernel_ir.to_json(QGEMV_TERM).encode()
        ).hexdigest()
    )
    assert len(manifest["changes"]) == 1
    assert manifest["artifacts"] == [
        {
            "artifact": manifest["changes"][0]["artifact"],
            "hash": manifest["changes"][0]["hash"],
        }
    ]
    assert manifest["verify"]["level"] == "term-denotation"
    assert manifest["verify"]["artifact_gate"] is None


def test_lower_candidate_rejects_stub_term(stub_bins) -> None:
    class StubTerm:
        pass

    cand = SimpleNamespace(term=StubTerm(), label="stub")
    with pytest.raises(LowerError, match=r"lib_kernel_ir\.Term"):
        lower_candidate(cand, "triton", bin_dir=stub_bins)


# ---------------------------------------------------------------------------
#  verify_lowered — the v1 (denotation-level) verify ceiling
# ---------------------------------------------------------------------------


def test_verify_lowered_redenotes() -> None:
    env = {"x": torch.ones(4)}
    out = verify_lowered(ELEMENTWISE_TERM, env)
    assert torch.equal(out, -torch.ones(4))


def test_verify_lowered_rejects_non_term() -> None:
    with pytest.raises(LowerError, match=r"lib_kernel_ir\.Term"):
        verify_lowered(object(), {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
#  Real emitters — skip honestly when no built binary resolves
# ---------------------------------------------------------------------------


def _real_binary(target: str):
    """A built emitter binary, or None — never triggers a cargo build."""
    spec_env = {
        "triton": "COMPILE_TRITON_BIN",
        "cuda": "COMPILE_CUDA_BIN",
    }
    saved = os.environ.get(spec_env[target])
    try:
        os.environ.pop(spec_env[target], None)
        os.environ.pop("KERNEL_LOWER_BIN", None)
        return lower_mod._find_binary(target, None)
    finally:
        if saved is not None:
            os.environ[spec_env[target]] = saved


@pytest.mark.parametrize("target", ["triton", "cuda"])
def test_lower_term_real_emitter(target) -> None:
    if _real_binary(target) is None:
        pytest.skip(f"no built {target} emitter binary — skipping")
    changes = lower_term(QGEMV_TERM, target)
    assert changes[0]["op"] == "put"
    assert (
        changes[0]["hash"]
        == blake3.blake3(
            base64.b64decode(changes[0]["content_b64"])
        ).hexdigest()
    )
    source = base64.b64decode(changes[0]["content_b64"]).decode()
    assert source.strip()  # the emitter produced something


def test_lower_term_real_emitter_refusal() -> None:
    """A term outside both fragments is refused by name, verbatim."""
    if _real_binary("triton") is None:
        pytest.skip("no built triton emitter binary — skipping")
    # `neg` of a leaf is elementwise — lowerable.  A bare `dot` with a
    # computed lhs is not in the qgemv spelling: build one the emitter
    # must refuse.
    bad = Term(
        "store",
        (
            Term("dot", (Term("neg", (Leaf("a"),)), Leaf("b"))),
            Leaf("out"),
        ),
    )
    with pytest.raises(LowerError, match="not a lowerable term"):
        lower_term(bad, "triton")
