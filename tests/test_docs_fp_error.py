"""docs/fp-error.md must stay true.

Every Python block marked ``<!-- doctest -->`` is executed, and the output the page quotes is
checked against what is really printed.
"""

import os
import re

import pytest

_DOC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "fp-error.md")
_BLOCK = re.compile(r"<!-- doctest -->\n```python\n(.*?)```", re.DOTALL)


def _blocks():
    with open(_DOC) as f:
        return _BLOCK.findall(f.read())


def test_doc_has_runnable_blocks():
    assert len(_blocks()) == 3


def test_doc_snippets_run_and_quoted_output_is_real(capsys):
    pytest.importorskip("onnxruntime")
    namespace: dict = {}
    for i, block in enumerate(_blocks()):
        try:
            exec(compile(block, f"docs/fp-error.md block {i}", "exec"), namespace)
        except Exception as e:  # say which block of the page broke
            raise AssertionError(
                f"docs/fp-error.md doctest block {i} failed: {e!r}"
            ) from e
    out = capsys.readouterr().out
    for line in (
        "fp32: certified |error| <= 3.1e-05",
        "fp16: certified |error| <= 2.7e-01",
        "bf16: certified |error| <= 2.8e+00",
        "no box: inf",
        "certified atol = 3.9e-05",
        "  real-arithmetic difference : 4.2e-07",
        "  roundoff of the original   : 2.3e-05",
        "  roundoff of the simplified : 1.5e-05",
        "interval ranges, forward only : 1.58e-02",
        "zonotope ranges, forward only : 1.37e-02",
        "+ tight pass                  : 7.84e-03",
    ):
        assert line in out, f"the page quotes {line!r} but it was not printed:\n{out}"
