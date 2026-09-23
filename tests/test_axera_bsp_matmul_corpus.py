"""Checks on four vendor-compiled MatMul models from Axera's public AX650 BSP.

The models are ``AXERA-TECH/ax650n_bsp_sdk``'s IVE matmul samples, compiled by
Axera's own toolchain rather than this project's pulsar2 7.0-lite. See
``docs/axera-bsp-mining.md``. No network or device access is needed.
"""

import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import bsp_matmul_corpus as corpus  # noqa: E402
import mcode  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures", "bsp_matmul")
NPU1_M16 = "v1_matmul_npu1_s8_256_16_10000_136009.axmodel.gz"
NPU1_M32 = "v1_matmul_npu1_s8_256_32_10000_161080.axmodel.gz"
NPU3_S8 = "v1_matmul_npu3_s8_512_16_10000_493999.axmodel.gz"
NPU3_S16 = "v1_matmul_npu3_s16_512_16_10000_1142387.axmodel.gz"


def _load(name):
    row = corpus.load(os.path.join(_FIX, name))
    row["gen"] = corpus.gen(name)
    return row


@pytest.mark.parametrize("name", [NPU1_M16, NPU1_M32, NPU3_S8, NPU3_S16])
def test_vendor_streams_pass_the_structural_validator(name):
    # A compiler-version cross-check: mcode.check() was derived from this
    # project's own pulsar2 7.0-lite builds.
    assert mcode.check(_load(name)["mcode"]) == []


def test_graph_is_two_live_quantized_inputs():
    row = _load(NPU3_S8)
    assert (row["dtype"], row["K"], row["N"], row["M"]) == ("s8", 512, 10000, 16)
    assert _load(NPU3_S16)["dtype"] == "s16"


def test_segment_count_is_five_per_core():
    assert len(corpus.segment_sizes(_load(NPU1_M16)["mcode"])) == 5
    assert len(corpus.segment_sizes(_load(NPU3_S8)["mcode"])) == 15


def test_s8_teng_is_a_small_fixed_template():
    npu3 = corpus.segment_sizes(_load(NPU3_S8)["mcode"])
    assert [npu3[i] for i in corpus.ENGINES["npu3"]["teng"]] == [544, 544, 544]
    assert corpus.segment_sizes(_load(NPU1_M16)["mcode"])[2] == 640


def test_teng_template_does_not_encode_m():
    # M=16 and M=32 at K=256, N=10000 compile to a byte-identical TENG queue.
    a = corpus.segment_bytes(_load(NPU1_M16)["mcode"], 2)
    b = corpus.segment_bytes(_load(NPU1_M32)["mcode"], 2)
    assert a == b
    # ...while the whole stream does differ.
    assert _load(NPU1_M16)["mcode"] != _load(NPU1_M32)["mcode"]


def test_s16_grows_teng_far_more_than_conv():
    s8 = corpus.segment_sizes(_load(NPU3_S8)["mcode"])
    s16 = corpus.segment_sizes(_load(NPU3_S16)["mcode"])
    eng = corpus.ENGINES["npu3"]

    def total(sizes, name):
        return sum(sizes[i] for i in eng[name])

    teng_ratio = total(s16, "teng") / total(s8, "teng")
    conv_ratio = total(s16, "conv") / total(s8, "conv")
    # 12.9x vs 8.1x for this shape; the corpus medians are in the doc.
    assert teng_ratio > conv_ratio


def test_output_row_stride_is_in_the_dma_queue():
    row = _load(NPU1_M16)
    dma = corpus.segment_bytes(row["mcode"], corpus.ENGINES["npu1"]["dma"][0])
    assert (row["N"] * 4).to_bytes(4, "little") in dma
