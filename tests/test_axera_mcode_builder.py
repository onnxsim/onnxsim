import os
import sys

import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import mcode_builder  # noqa: E402
import step_recalibrate  # noqa: E402


def test_builder_emits_decoded_record_stream():
    program = (
        mcode_builder.MCodeProgram()
        .verb(0xA1, 0x40, 0x02, b"\x01\x02\x03\x04")
        .short(1, b"\x98\x02", 0x83, 0x40)
        .bare(0x83, 0x0E)
        .raw(0)
    )
    assert program.encode() == bytes.fromhex(
        "a1 00 40 02 01 02 03 04 01 98 02 83 40 83 0e 00"
    )


def test_builder_rejects_invalid_widths_and_operands():
    with pytest.raises(ValueError, match="operand"):
        mcode_builder.MCodeProgram().verb(0xA1, 0, 0, b"\x00")
    with pytest.raises(ValueError, match="payload length"):
        mcode_builder.MCodeProgram().short(1, b"\x00", 0x83, 0)
    with pytest.raises(ValueError, match="extra byte"):
        mcode_builder.MCodeProgram().short(1, b"\x00\x00", 0x83, 0, b"\x01")


def test_builder_can_extend_a_decoded_stream():
    program = mcode_builder.MCodeProgram().extend(
        [{"kind": "raw", "byte": 0x42}, {"kind": "raw", "byte": 0x00}]
    )
    assert program.encode() == b"\x42\x00"


def test_builder_can_relayout_an_existing_template_segment():
    path = os.path.join(_AXERA, "fixtures", "step_recalib", "toyf_A1.axmodel.gz")
    template = step_recalibrate.get_mcode(step_recalibrate.load(path))
    raw = step_recalibrate.codec.decode_segments(template)[1]
    program = mcode_builder.MCodeProgram().extend(
        step_recalibrate.mcode_mod.decode(
            raw, start=0, end=len(raw), **step_recalibrate.mcode_mod.FULL_RULE
        )
    )
    assert mcode_builder.replace_template_segment(template, 1, program) == template


def test_builder_packages_a_segment_into_an_axmodel():
    path = os.path.join(_AXERA, "fixtures", "step_recalib", "toyf_A1.axmodel.gz")
    model = step_recalibrate.load(path)
    name = step_recalibrate.mcode_name(model)
    raw = step_recalibrate.codec.decode_segments(step_recalibrate.get_mcode(model))[1]
    program = mcode_builder.MCodeProgram().extend(
        step_recalibrate.mcode_mod.decode(
            raw, start=0, end=len(raw), **step_recalibrate.mcode_mod.FULL_RULE
        )
    )
    packaged = mcode_builder.replace_model_segment(model, name, 1, program)
    init = next(item for item in packaged.graph.initializer if item.name == name)
    assert bytes(init.raw_data) == bytes(
        next(item for item in model.graph.initializer if item.name == name).raw_data
    )
    assert list(init.dims) == [len(init.raw_data)]
