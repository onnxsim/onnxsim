"""Runs every ``<!-- doctest -->`` Python block of docs/quant-int-verify.md and checks that the
output quoted right after it is what the code really prints, so the doc cannot drift from the code."""

import contextlib
import io
import pathlib
import re

import pytest

from onnxsim import quant_int_verify as Q

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "quant-int-verify.md"
_BLOCK = re.compile(
    r"<!-- doctest -->\n```python\n(.*?)```\n```text\n(.*?)```", re.DOTALL
)


# Snippets whose output depends on values read back from onnxruntime (directly, or through
# replay_no_wrap, which replays the counterexample on it).
_USES_ORT = ("onnxruntime", "replay_no_wrap", "replay_")


def _blocks():
    return _BLOCK.findall(_DOC.read_text())


def test_the_doc_has_runnable_examples():
    assert len(_blocks()) >= 5


@pytest.mark.parametrize("index", range(len(_blocks())))
def test_doc_example_prints_what_the_doc_says(index):
    code, expected = _blocks()[index]
    if any(m in code for m in _USES_ORT) and Q.probe_u8s8_saturation() is True:
        # the quoted output includes values read back from onnxruntime; on a CPU whose u8xs8
        # kernel saturates int16 pair sums those differ from exact int32 arithmetic
        pytest.skip("this host's onnxruntime u8xs8 kernel saturates int16 pair sums")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, f"{_DOC.name}[{index}]", "exec"), {"__name__": "__doc__"})
    assert out.getvalue().strip() == expected.strip()
