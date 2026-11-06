"""Host-only checks of ``scripts/axera/pulsar2_docker.llm_build``'s dtype arguments.

The value sets are the ones ``pulsar2 llm_build`` 7.0 lists in its ``--help``
(hidden state, layer weights, post weights). Unknown values raise before any
Docker call, and every accepted value reaches the command line. No Docker,
no Pulsar2, no checkpoint: ``docker_image_available`` and ``subprocess.run``
are replaced.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

# ``pulsar2_docker`` imports the repo onnxsim at load (``ensure_repo_onnxsim``), which needs the
# compiled extension. ``llm_build`` never touches onnxsim, so the helper is stubbed here.
import types  # noqa: E402

sys.modules.setdefault(
    "_local_import", types.SimpleNamespace(ensure_repo_onnxsim=lambda: None)
)
import pulsar2_docker as P  # noqa: E402


class _Done:
    returncode = 1
    stdout = ""
    stderr = "stop after the command line was captured"


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Capture the ``docker run`` command; the build itself is not run."""
    calls = []
    monkeypatch.setattr(P, "docker_image_available", lambda image: True)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    return calls, tmp_path


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


@pytest.mark.parametrize("value", P.LLM_BUILD_HIDDEN_TYPES)
def test_hidden_state_types_reach_the_command(captured, value):
    calls, work = captured
    P.llm_build(str(work), "ckpt", "out", hidden_state_type=value)
    assert _arg(calls[0], "--hidden_state_type") == value


@pytest.mark.parametrize("value", P.LLM_BUILD_WEIGHT_TYPES)
def test_weight_types_reach_the_command(captured, value):
    calls, work = captured
    P.llm_build(str(work), "ckpt", "out", weight_type=value)
    assert _arg(calls[0], "--weight_type") == value


@pytest.mark.parametrize("value", P.LLM_BUILD_POST_TYPES)
def test_post_weight_types_reach_the_command(captured, value):
    calls, work = captured
    P.llm_build(str(work), "ckpt", "out", post_weight_type=value)
    assert _arg(calls[0], "--post_weight_type") == value


def test_post_weight_default_is_s8(captured):
    calls, work = captured
    P.llm_build(str(work), "ckpt", "out")
    assert _arg(calls[0], "--post_weight_type") == "s8"


def test_s4_is_not_a_post_weight_type(captured):
    # pulsar2 7.0 rejects `--post_weight_type s4` (checked in the container)
    calls, work = captured
    with pytest.raises(ValueError, match="post_weight_type"):
        P.llm_build(str(work), "ckpt", "out", post_weight_type="s4")
    assert calls == []


@pytest.mark.parametrize(
    "kw",
    [
        {"hidden_state_type": "int8"},
        {"weight_type": "int4"},
        {"weight_type": "S8"},
        {"post_weight_type": "fp16"},
    ],
)
def test_unknown_values_raise_before_docker(captured, kw):
    calls, work = captured
    with pytest.raises(ValueError):
        P.llm_build(str(work), "ckpt", "out", **kw)
    assert calls == []


def test_value_sets_match_pulsar2_help():
    # the sets as `pulsar2 llm_build --help` lists them (7.0-lite image)
    assert P.LLM_BUILD_HIDDEN_TYPES == ("fp16", "bf16", "fp32")
    assert P.LLM_BUILD_WEIGHT_TYPES == (
        "fp16",
        "bf16",
        "fp32",
        "s8",
        "s4",
        "fp8_e5m2",
        "fp8_e4m3",
    )
    assert P.LLM_BUILD_POST_TYPES == ("bf16", "s8", "fp8_e5m2", "fp8_e4m3")
