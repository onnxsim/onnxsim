"""docs/range-opt.md must stay true.

Every ``<!-- doctest -->`` Python block is executed (all blocks share one namespace, in order) and
the output the page quotes right after it must be exactly what the code prints.
"""

import contextlib
import io
import pathlib
import re

import pytest

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "range-opt.md"
_BLOCK = re.compile(
    r"<!-- doctest -->\n```python\n(.*?)```\n```text\n(.*?)```", re.DOTALL
)


def _blocks():
    return _BLOCK.findall(_DOC.read_text())


def test_the_doc_has_runnable_examples():
    assert len(_blocks()) == 5


def test_doc_examples_print_what_the_doc_says():
    namespace = {"__name__": "__doc__"}
    for i, (code, expected) in enumerate(_blocks()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"{_DOC.name}[{i}]", "exec"), namespace)
        assert out.getvalue().strip() == expected.strip(), (
            f"block {i} printed something else"
        )


@pytest.mark.parametrize("name", ["scripts/range_opt_bench.py"])
def test_the_benchmark_script_exists_and_parses(name):
    path = _DOC.parents[1] / name
    compile(path.read_text(), str(path), "exec")
