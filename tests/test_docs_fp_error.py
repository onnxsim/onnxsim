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
    # (quoted line, relative tolerance). Almost every figure is deterministic; the exception
    # is the real-arithmetic difference, which comes from float32 constants that simplify()
    # folds, and the folding arithmetic differs in the last digit across platforms (the page
    # quotes 4.2e-07; aarch64 CI prints 3.9e-07). The label must still match exactly.
    for line, rtol in (
        ("fp32: certified |error| <= 3.1e-05", 0.02),
        ("fp16: certified |error| <= 2.7e-01", 0.02),
        ("bf16: certified |error| <= 2.8e+00", 0.02),
        ("no box: inf", 0.0),
        ("certified atol = 3.9e-05", 0.05),
        ("  real-arithmetic difference : 4.2e-07", 0.25),
        ("  roundoff of the original   : 2.3e-05", 0.05),
        ("  roundoff of the simplified : 1.5e-05", 0.05),
        ("interval ranges, forward only : 1.58e-02", 0.02),
        ("zonotope ranges, forward only : 1.37e-02", 0.02),
        ("+ tight pass                  : 7.84e-03", 0.02),
    ):
        assert _quoted_line_is_printed(line, rtol, out), (
            f"the page quotes {line!r} (within {rtol:.0%}) but it was not printed:\n{out}"
        )


_NUMBER = re.compile(r"^(?P<label>.*?)(?P<num>-?(?:inf|\d+(?:\.\d+)?(?:e[+-]?\d+)?))$")


def _quoted_line_is_printed(quoted: str, rtol: float, out: str) -> bool:
    """True if ``out`` has a line with the quoted label whose number matches within ``rtol``."""
    q = _NUMBER.match(quoted)
    assert q, f"cannot parse the quoted line {quoted!r}"
    for got in out.splitlines():
        g = _NUMBER.match(got)
        if g and g.group("label") == q.group("label"):
            a, b = float(g.group("num")), float(q.group("num"))
            if a == b or abs(a - b) <= rtol * abs(b):
                return True
    return False
