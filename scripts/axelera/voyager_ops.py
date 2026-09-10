"""Metis AIPU operator-support data for Axelera's Voyager SDK
(https://github.com/axelera-ai-hub/voyager-sdk), scraped from its own public
docs -- see `scrape_onnx_support_docs.py` for how, and this module's
docstring for a crucial caveat about the actual `.axmodel` compiler this data
describes.

Unlike Axera's Pulsar2 (`scripts/axera/pulsar2_ops.py`), Voyager SDK does
publish a proper machine-readable-ish per-operator support reference
(`docs/reference/compiler/onnx-support.md` and its per-opset detail pages),
including formal `rule`/`allow_config` predicates for most "Constrained"
operators -- not just an op-type support list. `voyager_op_support_data.py`
(auto-generated, do not hand-edit) is a faithful transcription of that.

**This data itself is untouched by any compiler run** -- it's a literal
transcription of the docs, nothing more. But unlike an earlier version of
this docstring claimed, the real compiler *is* reachable here: `axelera-rt`/
`axelera-devkit` (providing `axelera.compiler`) install from a genuinely
public Artifactory PyPI mirror with no login (`docs/user-guides/sdk-
install.md`'s own documented command) -- the "proprietary, credentials-only"
claim was based on the deprecated installer's `axelera_runtime` package name
(`installer_support.py`) and was never re-tested against the current pip
path. See `voyager_backend.py` for the real quantizer wrapper, and its
docstring for a concrete finding that matters here: the real compiler's
error message for a rule this data records (`Conv`'s `auto_pad ==
"NOTSET"`) quotes that exact string back -- real, if narrow, confirmation
that this transcription matches the compiler's actual internal check, not
just its prose documentation. `voyager_simulator.py`'s docstring explains
what that does and doesn't license this module's own (still real-compiler-
free) `evaluate_constraints()` to claim.
"""

from voyager_op_support_data import VOYAGER_OP_SUPPORT

#: {op_type: 'Supported' | 'Constrained'} -- every operator Voyager SDK's
#: docs list at opset 17 (its own recommended default opset; see
#: `scrape_onnx_support_docs.py` for why only one opset is captured). An
#: op_type *not* in this dict is not documented as AIPU-accelerated at all
#: and, per onnx-support.md, "falls back to host CPU via ONNX Runtime
#: preamble/postamble" -- compilation still succeeds, just with that node
#: running on the host.
VOYAGER_OP_LEVEL = {name: entry["level"] for name, entry in VOYAGER_OP_SUPPORT.items()}

#: op_types documented as "Supported": accelerated with no attribute/shape
#: restrictions Voyager SDK's docs call out.
VOYAGER_UNCONSTRAINED_OPS = frozenset(
    name for name, level in VOYAGER_OP_LEVEL.items() if level == "Supported"
)

#: op_types documented as "Constrained": accelerated, but only for specific
#: attribute/shape configurations -- see VOYAGER_OP_SUPPORT[name]['rules'] /
#: ['allow_config'] / ['notes'].
VOYAGER_CONSTRAINED_OPS = frozenset(
    name for name, level in VOYAGER_OP_LEVEL.items() if level == "Constrained"
)

#: Constrained ops whose docs give no formal `rule`/`allow_config` predicate
#: at all -- only free-text "Axelera's notes for developers" (e.g. MatMul:
#: "supported only when it is part of the attention block used by the YOLO11
#: family of networks"). Nothing here is statically checkable from a node's
#: shape/constness alone; `voyager_simulator.py` always reports these as
#: unverified, never as violated.
VOYAGER_PROSE_ONLY_CONSTRAINTS = frozenset(
    name
    for name, entry in VOYAGER_OP_SUPPORT.items()
    if entry["level"] == "Constrained"
    and not entry["rules"]
    and not entry["allow_config"]
)
