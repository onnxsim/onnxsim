"""Tests for ``onnxsim._rich_compat`` -- the shim that makes ``rich`` optional.

``rich`` is only ever used to pretty-print terminal reports, so it lives in the
``onnxsim[rich]`` extra and the printing modules import ``print``/``Table``/
``Text`` from the shim instead of from ``rich`` directly. These tests cover both
halves of that: the plain-text stand-ins render something readable and
markup-free, and no module goes around the shim (which would reintroduce a hard
``rich`` dependency without anyone noticing).

The "rich is not installed" case is simulated -- the shim's fallback classes are
exercised directly, and the import-time branch is re-executed as a throwaway
module with ``rich`` made unimportable -- so the tests report the same result
whether or not ``rich`` happens to be installed.
"""

import builtins
import contextlib
import importlib.util
import pathlib
import re
import sys

import onnx.parser as parser
import pytest

from onnxsim import _rich_compat, memory_planning, model_info, onnx_simplifier
from onnxsim._rich_compat import _plain_print, _PlainTable, _PlainText, _strip_markup


@contextlib.contextmanager
def _rich_unimportable(monkeypatch):
    """Make ``import rich`` (and any submodule) raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("simulated: rich is not installed")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n == "rich" or n.startswith("rich.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield


def _load_shim_without_rich(monkeypatch):
    # A throwaway copy of the module, so the canonical onnxsim._rich_compat (and
    # the names model_info/memory_planning/onnx_simplifier already imported from
    # it) is left untouched.
    spec = importlib.util.spec_from_file_location(
        "_rich_compat_no_rich", _rich_compat.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with _rich_unimportable(monkeypatch):
        spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The shim is the only door to rich
# --------------------------------------------------------------------------- #
def test_printing_modules_import_from_the_shim():
    for module in (model_info, memory_planning):
        assert module.print is _rich_compat.print
        assert module.Table is _rich_compat.Table
        assert module.Text is _rich_compat.Text
    assert onnx_simplifier.print is _rich_compat.print
    assert onnx_simplifier.Text is _rich_compat.Text


def test_no_module_imports_rich_directly():
    # rich must stay optional: everything goes through _rich_compat, which is
    # the one module allowed to import it.
    package = pathlib.Path(model_info.__file__).parent
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if path.name != "_rich_compat.py"
        and re.search(r"^\s*(from|import) rich\b", path.read_text(), re.MULTILINE)
    ]
    assert offenders == []


# --------------------------------------------------------------------------- #
# The import-time fallback branch
# --------------------------------------------------------------------------- #
def test_shim_falls_back_when_rich_is_missing(monkeypatch, capsys):
    module = _load_shim_without_rich(monkeypatch)
    assert module.HAS_RICH is False
    assert module.print is module._plain_print
    assert module.Table is module._PlainTable
    assert module.Text is module._PlainText
    # The fallback names are usable exactly like rich's.
    table = module.Table(title="t")
    table.add_column("col")
    table.add_row(module.Text("value", style="bold green1"))
    module.print(table)
    assert "value" in capsys.readouterr().out


def test_shim_reports_rich_when_importable():
    rich = pytest.importorskip("rich")
    assert _rich_compat.HAS_RICH is True
    assert _rich_compat.print is rich.print


# --------------------------------------------------------------------------- #
# Markup stripping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("[bold]Nodes removed (2):[/bold]", "Nodes removed (2):"),
        ("[bold magenta]too large[/bold magenta]", "too large"),
        ("[dim]... and 3 more[/dim]", "... and 3 more"),
        ("[bold green1]4.0 MiB[/bold green1]", "4.0 MiB"),
        ("[#ff00aa]x[/#ff00aa] [color(3)]y[/color(3)]", "x y"),
        ("[not bold]x[/]", "x"),
        # Not markup: ordinary bracketed output must survive untouched.
        ("shape [1, 3, 224, 224]", "shape [1, 3, 224, 224]"),
        ("path C:\\models[v2]\\a.onnx", "path C:\\models[v2]\\a.onnx"),
        ("node [id0] removed", "node [id0] removed"),
        ("empty []", "empty []"),
    ],
)
def test_strip_markup(text, expected):
    assert _strip_markup(text) == expected


# --------------------------------------------------------------------------- #
# The plain-text stand-ins
# --------------------------------------------------------------------------- #
def test_plain_text_keeps_text_and_drops_style():
    text = _PlainText("hello", style="bold red")
    assert str(text) == "hello"
    assert text.plain == "hello"
    assert len(text) == 5
    text.append(_PlainText(" world"))
    assert str(text) == "hello world"


def test_plain_table_renders_an_ascii_grid():
    table = _PlainTable(title="Activation Memory Plan")
    table.add_column("Tensor")
    table.add_column("Offset")
    table.add_row("input", "0")
    table.add_row("a_much_longer_name", _PlainText("1.2 MiB", style="dim"))
    lines = str(table).splitlines()

    assert lines[0].strip() == "Activation Memory Plan"
    assert lines[1].startswith("+") and lines[1].endswith("+")
    assert [c.strip() for c in lines[2].strip("|").split("|")] == ["Tensor", "Offset"]
    assert [c.strip() for c in lines[4].strip("|").split("|")] == ["input", "0"]
    # Every row is padded to the same width, so the grid lines up.
    assert len({len(line) for line in lines[1:]}) == 1
    # Styles are dropped, not rendered as markup or escape codes.
    assert "1.2 MiB" in lines[5]
    assert "\x1b[" not in str(table)


def test_plain_table_without_title_or_rows():
    table = _PlainTable()
    table.add_column("only")
    assert str(table).splitlines() == ["+------+", "| only |", "+------+", "+------+"]


def test_plain_table_tolerates_short_and_long_rows():
    table = _PlainTable()
    table.add_column("a")
    table.add_column("b")
    table.add_row("1")
    table.add_row("2", "3", "4")
    rendered = str(table)
    for cell in ("1", "2", "3", "4"):
        assert cell in rendered


def test_plain_print_strips_markup_and_renders_objects(capsys):
    table = _PlainTable()
    table.add_column("col")
    table.add_row("v")
    _plain_print("[bold]title[/bold]", _PlainText("t", style="dim"), sep=" | ")
    _plain_print(table)
    _plain_print()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "title | t"
    assert "| v" in out
    assert "[bold]" not in out


# --------------------------------------------------------------------------- #
# The real report functions, with the stand-ins in place of rich
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_rich(monkeypatch):
    """Point the printing modules at the plain-text stand-ins, i.e. run them as
    they would run in an install without ``rich``."""
    for module in (model_info, memory_planning, onnx_simplifier):
        monkeypatch.setattr(module, "print", _plain_print, raising=False)
        monkeypatch.setattr(module, "Text", _PlainText, raising=False)
        monkeypatch.setattr(module, "Table", _PlainTable, raising=False)


def _model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 23]>
        g (float[1,4] x) => (float[1,4] y)
        {
          [id0] mid = Identity(x)
          [relu0] y = Relu(mid)
        }
        """
    )


def _simplified():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 23]>
        g (float[1,4] x) => (float[1,4] y)
        {
          [relu0] y = Relu(x)
        }
        """
    )


def test_print_simplifying_info_without_rich(no_rich, capsys):
    model_info.print_simplifying_info(_model(), _simplified())
    out = capsys.readouterr().out
    assert "Model Size" in out
    assert "Original Model" in out
    assert "Simplified Model" in out
    assert "[bold" not in out


def test_print_graph_diff_without_rich(no_rich, capsys):
    model_info.print_graph_diff(_model(), _simplified())
    out = capsys.readouterr().out
    assert "Nodes removed" in out
    assert "id0" in out
    # The section headings are markup in the rich path; they must not leak.
    assert "[bold]" not in out and "[/bold]" not in out


def test_print_memory_plan_without_rich(no_rich, capsys):
    plan = memory_planning.plan_activation_memory(_model())
    memory_planning.print_memory_plan(plan)
    out = capsys.readouterr().out
    assert "Activation Memory Plan" in out
    assert "Arena:" in out
    assert "[dim]" not in out
