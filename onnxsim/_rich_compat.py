"""``rich``-optional console output.

onnxsim uses ``rich`` for exactly three things -- its ``print``, its ``Table``
and its ``Text`` -- and only ever to make terminal reports (the
original-vs-simplified table, the memory plan, the graph diff, the weight
quantization error table, CLI warnings) look nicer. None of it affects the
models onnxsim produces, so ``rich`` is an *optional* dependency
(``pip install onnxsim[rich]``) rather than something every install has to
carry.

Modules that print should import ``print``/``Table``/``Text`` from here instead
of from ``rich``. When ``rich`` is installed these names *are* rich's own; when
it is not, they are the plain-text stand-ins below, which cover the small API
surface onnxsim actually uses:

- ``print(*objects)`` -- writes to stdout, rendering the stand-in ``Table`` and
  ``Text`` objects and stripping rich console markup (``[bold]...[/bold]``)
  from plain strings so tags never leak into the output.
- ``Table(title=...)`` with ``add_column``/``add_row`` -- rendered as an ASCII
  grid (deliberately ASCII: without rich there is no terminal detection to tell
  us whether box-drawing characters are safe to emit).
- ``Text(text, style=...)`` -- keeps the text, drops the style.

The stand-ins are intentionally not a rich re-implementation: no styling, no
wrapping, no width detection. They exist so that output stays readable, not so
that it stays identical.
"""

from __future__ import annotations

import builtins
import re
from typing import IO, Any, List, Optional

__all__ = ["HAS_RICH", "Table", "Text", "print"]


# --------------------------------------------------------------------------- #
# Markup stripping
# --------------------------------------------------------------------------- #
# Rich console markup is `[style]text[/style]`. Stripping every bracketed word
# would mangle ordinary output (tensor shapes, file paths, Python reprs), so a
# tag is only removed when *every* word in it names a style -- an attribute
# below, or a color (optionally `bright_`-prefixed and/or numbered, as in
# `green1`), a `#rrggbb` triplet, or `color(N)`.
_STYLE_ATTRS = frozenset(
    {
        "b",
        "blink",
        "blink2",
        "bold",
        "c",
        "conceal",
        "d",
        "default",
        "dim",
        "encircle",
        "frame",
        "i",
        "italic",
        "none",
        "not",
        "o",
        "on",
        "overline",
        "r",
        "reverse",
        "s",
        "strike",
        "u",
        "underline",
        "underline2",
        "uu",
    }
)
_COLOR_RE = re.compile(
    r"(?:bright_)?"
    r"(?:black|red|green|yellow|blue|magenta|cyan|white|grey|gray|purple)\d*$"
    r"|#[0-9a-fA-F]{6}$"
    r"|color\(\d+\)$"
)
_TAG_RE = re.compile(r"\[/?([a-zA-Z#()_ \d]*)\]")


def _is_style(word: str) -> bool:
    return word in _STYLE_ATTRS or _COLOR_RE.match(word) is not None


def _strip_markup(text: str) -> str:
    """Remove rich console markup tags from ``text``, leaving everything that
    is not a style tag (``[1, 2]``, ``C:\\[tmp]``, ...) untouched."""

    def replace(match: "re.Match[str]") -> str:
        words = match.group(1).split()
        # `[/]` (a bare close-all tag) has no words; `[]` is not markup.
        if not words:
            return "" if match.group(0) == "[/]" else match.group(0)
        return "" if all(_is_style(w) for w in words) else match.group(0)

    return _TAG_RE.sub(replace, text)


# --------------------------------------------------------------------------- #
# Plain-text stand-ins
# --------------------------------------------------------------------------- #
class _PlainText:
    """Stand-in for ``rich.text.Text``: carries the text, ignores the style."""

    def __init__(self, text: str = "", style: Any = "", **kwargs: Any) -> None:
        self.plain = str(text)
        self.style = style

    def append(self, text: Any, style: Any = None) -> "_PlainText":
        self.plain += _plain(text)
        return self

    def __len__(self) -> int:
        return len(self.plain)

    def __str__(self) -> str:
        return self.plain

    def __repr__(self) -> str:
        return f"Text({self.plain!r})"


def _plain(obj: Any) -> str:
    """Render one printable object as plain text."""
    if isinstance(obj, _PlainText):
        return obj.plain
    if isinstance(obj, str):
        return _strip_markup(obj)
    return str(obj)


class _PlainTable:
    """Stand-in for ``rich.table.Table``: an ASCII grid with an optional title.

    Only the ``add_column``/``add_row`` subset onnxsim uses is supported; every
    other rich ``Table`` keyword (styling, box, padding, ...) is accepted and
    ignored so callers do not have to branch on whether rich is installed.
    """

    def __init__(self, *headers: Any, title: Optional[Any] = None, **kwargs: Any):
        self.title = None if title is None else _plain(title)
        self.columns: List[str] = [_plain(h) for h in headers]
        self.rows: List[List[str]] = []

    def add_column(self, header: Any = "", **kwargs: Any) -> None:
        self.columns.append(_plain(header))

    def add_row(self, *cells: Any, **kwargs: Any) -> None:
        self.rows.append([_plain(c) for c in cells])

    def _widths(self) -> List[int]:
        width = max([len(self.columns)] + [len(r) for r in self.rows] + [0])
        widths = [0] * width
        for row in [self.columns] + self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        return widths

    def __str__(self) -> str:
        widths = self._widths()
        if not widths:
            return self.title or ""

        def line(row: List[str]) -> str:
            cells = row + [""] * (len(widths) - len(row))
            return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

        border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        lines = [border, line(self.columns), border]
        lines.extend(line(row) for row in self.rows)
        lines.append(border)
        if self.title is not None:
            lines.insert(0, self.title.center(len(border)).rstrip())
        return "\n".join(lines)


def _plain_print(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[IO[str]] = None,
    flush: bool = False,
    **kwargs: Any,
) -> None:
    """Stand-in for ``rich.print``: ``builtins.print`` plus markup stripping and
    rendering of the stand-in ``Table``/``Text`` objects."""
    builtins.print(
        *(_plain(o) for o in objects), sep=sep, end=end, file=file, flush=flush
    )


try:
    from rich import print as print  # noqa: F401
    from rich.table import Table as Table  # noqa: F401
    from rich.text import Text as Text  # noqa: F401

    HAS_RICH = True
except ImportError:  # rich is optional; fall back to the stand-ins above.
    print = _plain_print  # type: ignore[assignment]
    Table = _PlainTable  # type: ignore[assignment, misc]
    Text = _PlainText  # type: ignore[assignment, misc]

    HAS_RICH = False
