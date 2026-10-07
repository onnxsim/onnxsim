"""Runs the example in docs/quant-verify.md so the quoted output cannot rot.

The doc's python block is executed and every certified/observed number it prints is compared with the
number the doc quotes (relative tolerance 5%: the quantizer's calibration runs in onnxruntime float32, so
the last digits are not portable across platforms/versions, but the order of magnitude and the
relationships the prose claims are).
"""

import contextlib
import io
import pathlib
import re

import pytest

pytest.importorskip("onnxruntime")

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "quant-verify.md"


def _block(marker, lang):
    text = _DOC.read_text()
    m = re.search(rf"<!-- {marker} -->\n```{lang}\n(.*?)```", text, re.S)
    assert m, f"no {marker} block in {_DOC}"
    return m.group(1)


_NUM = re.compile(r"-?\d+\.?\d*(?:e[+-]?\d+)?")


def _numbers(line):
    line = re.sub(
        r"_v_\d+", "_v_", line
    )  # tensor names come from the quantizer, not from the claim
    return [float(x) for x in _NUM.findall(line)]


def test_doc_example_runs_and_matches_the_quoted_output():
    code = _block("doctest", "python")
    quoted = _block("doctest-output", "text").splitlines()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(_DOC), "exec"), {"__name__": "doc_example"})
    got = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(got) == len(quoted), "\n".join(got)
    for g, q in zip(got, quoted):
        if g.startswith("---"):
            assert g == q
            continue
        assert _NUM.sub("#", re.sub(r"_v_\d+", "_v_", g)) == _NUM.sub(
            "#", re.sub(r"_v_\d+", "_v_", q)
        ), (g, q)  # same words, only the numbers may differ
        gn, qn = _numbers(g), _numbers(q)
        assert len(gn) == len(qn), (g, q)
        for a, b in zip(gn, qn):
            assert a == pytest.approx(b, rel=0.05, abs=1e-6), (g, q)


def test_doc_claims_hold_in_the_executed_example():
    code = _block("doctest", "python")
    buf = io.StringIO()
    ns = {"__name__": "doc_example"}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(_DOC), "exec"), ns)
    narrow, wide = ns["report"], ns["wide"]
    assert narrow.within(100.0) and not narrow.within(10.0)
    assert not any(s.clipped for s in narrow.sites) and any(
        s.clipped for s in wide.sites
    )
    assert 20 < wide.worst / narrow.worst < 80  # "grows by a factor of 40"
