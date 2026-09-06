"""Tests for ``onnxsim.apply_attention_head_wanda_pruning_cpp`` -- the
C++-backed port of ``onnxsim.apply_attention_head_wanda_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s ``ApplyAttentionHeadWandaPruning``
and its own "Wanda calibration"/"Attention-head pruning" section comments).
The calibrated upgrade of ``onnxsim.apply_attention_head_pruning_cpp``,
exactly as ``onnxsim.apply_structured_wanda_pruning_cpp`` is to
``onnxsim.apply_structured_pruning_cpp`` -- runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``, the same executor
:func:`onnxsim.simplify` itself uses) to capture per-channel activation
norms right where each matched block's output projection reads them --
never a fake/mock executor.

Same chain-finding scope as ``onnxsim.apply_attention_head_pruning_cpp``
(plain ``com.microsoft::Attention``, ``com.microsoft::GroupQueryAttention``,
the plain ``ai.onnx::Attention`` op, ``com.microsoft::MultiHeadAttention``,
``com.microsoft::PackedMultiHeadAttention``,
``com.microsoft::DecoderMaskedMultiHeadAttention``,
``com.microsoft::PagedAttention``, the plain ``ai.onnx::LinearAttention`` op,
``com.microsoft::SparseAttention``, and the tenth, decomposed (un-fused,
"eager SDPA export") shape -- ``FindDecomposedGqaChains``/
``ApplyOneDecomposedGqaChain``, threaded through with a real calibrated
``act_norm`` map here exactly like every other family), minus the fused
``com.microsoft::MatMulNBitsQkv`` variant that port also matches -- this
Wanda port has no quantized-weight counterpart, mirroring the pure-Python
``onnxsim.apply_attention_head_wanda_pruning`` exactly (see
``structured_pruning_entry.h``'s own ``ApplyAttentionHeadWandaPruning``
declaration comment). Every family's own optional per-head bias/mask/
past-KV/head_sink/norm/scale input -- ``attention_bias``/``attn_mask``,
``past_key``/``past_value``, ``k_scale``/``v_scale``, ``head_sink``,
``q_norm_weight``/``k_norm_weight`` -- shares the identical
``HeadBiasInputIsSafe``/``SliceOrGatherHeadBias``/
``PastKvConstantsAreSliceable`` validate-and-slice machinery
``ApplyAttentionHeadPruning`` itself uses (``ApplyOneGqaChain``/
``ApplyOnePlainAttentionChain`` are shared verbatim between the two entry
points, just with a real calibrated ``act_norm`` map threaded through here
instead of ``nullptr``) -- see ``test_attention_head_pruning_cpp.py``'s own
docstring for the exact per-family scope. For the decomposed-GQA family, this
port's
``ApplyAttentionHeadWandaPruning`` shares ``FindDecomposedGqaChains``/
``ApplyOneDecomposedGqaChain`` entirely with ``ApplyAttentionHeadPruning``
(both dispatch through the identical matcher/rewriter, just with a real
calibrated ``act_norm`` map threaded through here instead of ``nullptr``) --
so every sub-shape that port matches (additive mask via
``HeadBiasInputIsSafe``/``SliceOrGatherHeadBias``, ``Einsum``-based QK^T/AV
products, decomposed RoPE/Q-K-norm pass-through, a packed-QKV-then-`Split`
producer, and the true-MQA fast path) is matched and pruned here too,
exactly mirroring pruning.py's own ``apply_attention_head_wanda_pruning``
for this one chain family -- see ``test_attention_head_pruning_cpp.py``'s
own docstring for the exact narrower-than-pruning.py scope each still
carries.

``onnxsim.apply_attention_head_wanda_pruning`` (the pure-Python name) is now
itself a thin alias for this port (full parity verified -- see pruning.py's
own "Attention-head pruning" section comment), exactly like
``apply_attention_head_pruning``/``apply_wanda_pruning`` before it -- see
``test_attention_head_pruning_cpp.py``'s own docstring for the one
pre-existing, pre-approved narrowing (``com.microsoft::LinearAttentionGate``)
that survives the alias. Every test below that used to call BOTH entry
points and compare their live outputs would be tautological (literally the
same code path twice) if left as-is -- those now instead compare the C++
port's output against a golden fixture (`_GOLDEN`, base64-encoded serialized
``ModelProto`` bytes) captured from the real pure-Python implementation
*before* its own mutating body was replaced by the alias, mirroring
``test_wanda_pruning_cpp.py``'s own established convention.
"""

import base64
import os

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _golden(b64):
    return onnx.load_from_string(base64.b64decode(b64))


# Frozen from the real pure-Python reference's own output, on the exact
# model + arguments each corresponding test below builds, before that
# implementation's own mutating body was replaced by the thin C++ alias
# (see this file's own module docstring) -- mirrors
# `test_wanda_pruning_cpp.py`'s/`test_transformer_block_pruning_cpp.py`'s own
# established golden-fixture convention.
_GOLDEN = {
    "test_cpp_attention_head_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference": (
        "CAo68gwKtQEKDUF0dGVudGlvbkJpYXMKM0F0dGVudGlvbkJpYXMvYXR0ZW50aW9uX2hlYWRfcHJ1"
        "bmluZ19nYXRoZXJfaW5kaWNlcxItQXR0ZW50aW9uQmlhcy9hdHRlbnRpb25faGVhZF9wcnVuaW5n"
        "X2dhdGhlcmVkGitBdHRlbnRpb25CaWFzL2F0dGVudGlvbl9oZWFkX3BydW5pbmdfZ2F0aGVyIgZH"
        "YXRoZXIqCwoEYXhpcxgBoAECCpABCgFYCgRXcWt2CgRCcWt2CgAKAAotQXR0ZW50aW9uQmlhcy9h"
        "dHRlbnRpb25faGVhZF9wcnVuaW5nX2dhdGhlcmVkEgNjdHgiCUF0dGVudGlvbioQCgludW1faGVh"
        "ZHMYAqABAiobChBxa3ZfaGlkZGVuX3NpemVzQAhACEAIoAEHOg1jb20ubWljcm9zb2Z0ChgKA2N0"
        "eAoEV291dBIBWSIGTWF0TXVsOgASAWcqjwYICAgYEAFCBFdxa3ZKgAbVCIk9jGzrPZSsFEALCmM/"
        "nUwpP9PzIb91Wce+nxnEv6xkv79MwZa/Fu/Hv6xFrb9gSErA6+BtvysJlz8UYMg9NTBRP6Vhtj/o"
        "XJw/FTTsvfeO+73SVwrAc++8PxAyL8DiRbA/Bn1AwBxTPb9AX5M/siy9vv9j0T+t1gs/RHlkPnll"
        "Dr8Dfpg/qTSSP/GHpz+GcAfAEs/XvnY+QL9vEdQ/LwEWQC3dtb9KUoO9CZ+KPwn+V7+cpME9D6oi"
        "v4QE9r3sGh4/es7avqrg3L/aJ9+/tkjPvm/+Az7RTrw/p4ajvqvzIT++hlA/X59hvnZ+cD4C5g48"
        "vKYsPaQfVb0UQ30/lJiqvn86hL9FObe/ayguv7QWwz4B8lU/4Mh4P3B3GL7/zp0/S/GBvxZItj7D"
        "ueS9WmfXvjHmsL+u+Nw9lqw+v8tb8D6ikRo+a4cXP/znLr+C8NW/HGT0v9rnPr++h6Q/kQczPb/X"
        "274Eii5AMdMWP2+YQ79/APU+FzExv6puzL8anIC/nLuTvx1sLz+Xf0zABSfzPRQJyL8x7rQ+52Dq"
        "P4o8hL+CkII/FUuvP/pMq79iZCw/i/vnPmpZtz8GA56/4/GFv1aCUr9IKXg+BuheP4ThjD4Qzr8/"
        "iifyvuOyib8xzGg/1kFJP+/C5z9b2UO+zL2/PpQBNb+WBbg+bVrcvip2tb1dQHu/BtbAv5ODtL0k"
        "sAw9mP5ZP+CKGT+BquW/U4nMv4qtwr7p4fW+g6NgP1Wz7z1U2DY/UEt6vw+bNz8ZSwjAakCsv2ke"
        "pD+GiOy+oMp9P0VGSL8M/oi/UeWWv2kIoD/1DpK/IDdXPzr79z7IvJI/u7buPyGrtD7JQhE/GRNX"
        "P+BC677KZuI/+Mmdvc/1d7/mFJU/5bq9P7BXFT/JVrA+W5ooPkvubr8QMDW/iJjZPhhS6j/b//k+"
        "aIAKvj1FTj+bvEe/sR5nPxZfK79hefk91BFZP6JF+D/T2r+/m9aSvsRb3D6kvglAGSiFP/G9LkBv"
        "dAA+lbJFv4v+y74qzwEICAgGEAFCBFdvdXRKwAHBmsA/jGYAQBRskj/SErI+kjlBvz22Tb+k9+Y+"
        "UZK1vmzA7D+8JHi/gl1qvkL8Xz++CKC+Ju3Av+eg9T1x2lE/ruT2vse1h7+Npp6+SXNqvx5Pmz+0"
        "y0Q9RI+XPtDzIj/5XYi+gWwrvyRgZ718ywvA1F+TP0QhJD+CRci+WeYHPcb5Cr/iLVa+sWTTvo7J"
        "o759ray+WDw6Ph3UAz4SQ5A/RI/TvE8+tj/inYa+pFkCPzYGET6Sjp89cE5Ov1s+Ij8qbAgYEAFC"
        "BEJxa3ZKYPhUkT/asMG+EBjkPvZ5Vj3JWTk/tBUgPxF237+D2/u+U06QvjowgL+9Tx2+jc3VvjoU"
        "Nr/60gw9FDCXv/Y+075Jz0Q+n0+2PuHkeD63sZY/HVd4P4fPfT/Cn02+LeryPypLCAIQB0IzQXR0"
        "ZW50aW9uQmlhcy9hdHRlbnRpb25faGVhZF9wcnVuaW5nX2dhdGhlcl9pbmRpY2VzShAAAAAAAAAA"
        "AAMAAAAAAAAAWh8KAVgSGgoYCAESFAoHEgViYXRjaAoFEgNzZXEKAggIWicKDUF0dGVudGlvbkJp"
        "YXMSFgoUCAESEAoCCAEKAggECgIIBQoCCAViHwoBWRIaChgIARIUCgcSBWJhdGNoCgUSA3NlcQoC"
        "CAZCBAoAEBFCEQoNY29tLm1pY3Jvc29mdBAB"
    ),
    "test_cpp_attention_head_wanda_pruning_matches_python_reference_multi_batch": (
        "CAo6jgUKXQoBWAoEV3FrdgoEQnFrdhIDY3R4IglBdHRlbnRpb24qEAoJbnVtX2hlYWRzGAKgAQIq"
        "GwoQcWt2X2hpZGRlbl9zaXplc0AEQARABKABBzoNY29tLm1pY3Jvc29mdAoYCgNjdHgKBFdvdXQS"
        "AVkiBk1hdE11bDoAEgFnKq8CCAYIDBABQgRXcWt2SqAC69Pfva3oBT+g5OA/lrWZPw+E8T2RV8s/"
        "ArAkPzxLGr+c/Qw+JeEWwDIdAb+Uhfi/AToQv+Lhi7/m6LK/3Dagv9l0sb86U5a+QrisvsJ4iT/j"
        "nFI/DmghwMX9KT6wvDi/5spRv8Iw1r6duCRAAwE9v4lHrr9JdMo+eE6Tv6n6qb5eIc0+p/VUvL0/"
        "kz78MYG/04A8QBDUyT7uJjq/lTCDP84zyj8fCWU+QeBMPzY6sD/bDrU+qdS/v+uVOr7n8bY/hx5d"
        "v60FaD4z0xS/B1W8Psb3lr+05W2/bv3fPIEWMb9dCHa/f3gKv8Qrbr4QhqO/Z4jIv/0M0j+GFtM+"
        "OQGEv6Wajz9KaLK/p4MWv9enCL8HJfc/tZtHP2I4uz5yBCO/Kl4IBAgFEAFCBFdvdXRKUOsZlL85"
        "RKq/BoWVPsbgCj9EqiNAVrvdvhOvYT9aYXc+kAtOP+Fvc7686xO/K/vlPYOTRD9lpi2/Jq6fPend"
        "EL8Cv4M+kVynPtoVz75GX+o/KjwIDBABQgRCcWt2SjCytja/TsSsPrwHGEBPgoI/MB0JP3CKGT8L"
        "7WY/VB5FP/qyFb+BipA/CtCpvkZKzL9aHwoBWBIaChgIARIUCgcSBWJhdGNoCgUSA3NlcQoCCAZi"
        "HwoBWRIaChgIARIUCgcSBWJhdGNoCgUSA3NlcQoCCAVCBAoAEBFCEQoNY29tLm1pY3Jvc29mdBAB"
    ),
    "test_cpp_attention_head_wanda_pruning_matches_python_reference_plain": (
        "CAo6kAoKXQoBWAoEV3FrdgoEQnFrdhIDY3R4IglBdHRlbnRpb24qEAoJbnVtX2hlYWRzGAKgAQIq"
        "GwoQcWt2X2hpZGRlbl9zaXplc0AIQAhACKABBzoNY29tLm1pY3Jvc29mdAoYCgNjdHgKBFdvdXQS"
        "AVkiBk1hdE11bDoAEgFnKo8GCAgIGBABQgRXcWt2SoAGU+pdP3PhlD64Q22/2HB8v95b2T7INyy/"
        "ELBtP1WjAr+EtNo/ivICQF+E6r43+2U/iLzsvh/+AL40xFm/kQ+8PoSSQj87FrM/7bB7vH78fz+q"
        "zPw9GGCXv3XcY74lZ7k/ZOYcv5CIU758GNQ+vXmYv9lF4D+WUss/2tAcP6jzH7765E0+4QdZvzPk"
        "ab/6G/Y/M1lHPlwktr6mOp49w+2mvpMFL7219Vs+q5Yav+efPD4Q0W0+SX3Jv+xukz52PeK/Ce/d"
        "vgxz4L5wZyjAqlCpP+bCwL3lyx3AgjrsP6IQ0T8V7bY/05ZuPs8lrb+DGP49GPFJP0q/3j/5mo++"
        "sz7FPRqO3j+ODhk+8pudPj2VSb7uR0A+evIcQAlPbT/yIJK+p4VGP8pmLT0Blfc+NDd6Pyuz8b6v"
        "YXbAwercv+vTB7+SYyDALPLkv9ftfr8GjWa/MepuPgWfhT9Bscy+2YUIQBBKkL+XAzA/xDehPePH"
        "GUA7ow2+DnBHwPOQ77+IjYC/l9UXv4+UiL2s63C/5b4xvVP0NL96ixO/EAuKP6buAEARk7y/8xxZ"
        "PvJLAT9HoqK/0QkUvjQYoT5/HDI/g2sbP4lwoj71ZMg+gZRtv0opC79hQg2/Fm4/Pwya8j8b1lq/"
        "WatJPqpDeD9I/IA+L2kpvxY66z4axXC+cT5dv57cmr9m9pG/AwITvp3f8L+e59g/+J5sP4CSob/+"
        "Kpa9uRSLPzGjeT9ejpa+4eQmQK77oj9m89O/GAvRvdGWiz896Qm/lBkGPzAFgr8wS7g/g22Sv0yK"
        "0b1/tA6/PjDpPuXr+r/F4B6/QaTBPzp5XT8eDnI/DBYyv6sfPD8phsQ9JbNeP45Kjj8XNWi/DhRj"
        "P0FjsL4/UwM/BWu7PjEgiD/dmDk/59AKPs6KGD9RWKC/jZa4v0D4pL+Q/M8/n9wMv9LHfD4+q7G/"
        "ZF5DPSTLZT+4+gfA1uSePyaljz4CRqE+F+Q/vsy/JL/uhkO/vN2bvkWBw78xPro/t8DzPtgDTb40"
        "LyQ+Ks8BCAgIBhABQgRXb3V0SsABUTVaO3Azir9Ltvm+9Cb/P1qpFD9pATa/bNOxvjGQLz2OaFO9"
        "ZPh2v8F9ZD89w4i+IqqAvzU2JT2Y5Ti/roSOP+jJnL90tLo/i7mqvn2NVL9iOwE+gZHbvaCUUT+H"
        "Mg6/cCSAP1u9bL+MPa8/vQp6PXB4iL8ojd0/u87ivozcF787Phm/n3baP501vL60X48/W4SaP76u"
        "yr/uYd8/abmOP60H6D4FDbE/Hgrhv8XvUL9RpL0/NLjZv5o+/r9SK6S/KmwIGBABQgRCcWt2SmBP"
        "1IE/utnsPVDoyz+k5xy/FDlEPwcM2z9xzrE+l7CJPjhQZj/LkB4+aCaNvy6neT347dW/So0IQO/M"
        "qj94Rxi/cbP+P5dbKL/XcY4+fd49P7JR9T9uXiW/a2A9P9XYCD9aHwoBWBIaChgIARIUCgcSBWJh"
        "dGNoCgUSA3NlcQoCCAhiHwoBWRIaChgIARIUCgcSBWJhdGNoCgUSA3NlcQoCCAZCBAoAEBFCEQoN"
        "Y29tLm1pY3Jvc29mdBAB"
    ),
    "test_cpp_attention_head_wanda_pruning_per_head_attention_bias_matches_python_reference": (
        "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"
    ),
    "test_cpp_decomposed_gqa_wanda_pruning_clones_shape_constant_shared_with_foreign_reader": (
        "CAo6kX4KHgoBWAoKWEZsYXRTaGFwZRICeGYiB1Jlc2hhcGU6AAoYCgJ4ZgoCV3EKAkJxEgJxMCIE"
        "R2VtbToACh4KAnEwCglTcV9wcnVuZWQSAnFyIgdSZXNoYXBlOgAKKAoCcXISAnF0IglUcmFuc3Bv"
        "c2UqEQoEcGVybUAAQAJAAUADoAEHOgAKIAoCeGYKAlNxEgtmb3JlaWduX291dCIHUmVzaGFwZToA"
        "ChgKAnhmCgJXawoCQmsSAmswIgRHZW1tOgAKGAoCazAKA1NrdhICa3IiB1Jlc2hhcGU6AAoYCgJ4"
        "ZgoCV3YKAkJ2EgJ2MCIER2VtbToAChgKAnYwCgNTa3YSAnZyIgdSZXNoYXBlOgAKKQoCa3ISA2t0"
        "MCIJVHJhbnNwb3NlKhEKBHBlcm1AAEACQAFAA6ABBzoAChsKA2t0MAoDQXgyEgJrdSIJVW5zcXVl"
        "ZXplOgAKIAoCa3UKDEtFeHBhbmRTaGFwZRICa2UiBkV4cGFuZDoACiEKAmtlCgtLTWVyZ2VTaGFw"
        "ZRIDa3JlIgdSZXNoYXBlOgAKKQoDa3JlEgJrdCIJVHJhbnNwb3NlKhEKBHBlcm1AAEABQANAAqAB"
        "BzoACikKAnZyEgN2dDAiCVRyYW5zcG9zZSoRCgRwZXJtQABAAkABQAOgAQc6AAobCgN2dDAKA0F4"
        "MhICdnUiCVVuc3F1ZWV6ZToACiAKAnZ1CgxWRXhwYW5kU2hhcGUSAnZlIgZFeHBhbmQ6AAogCgJ2"
        "ZQoLVk1lcmdlU2hhcGUSAnZ0IgdSZXNoYXBlOgAKFgoCcXQKAmt0EgJxayIGTWF0TXVsOgAKGgoC"
        "cWsKBVNjYWxlEgZzY2FsZWQiA011bDoACi8KBnNjYWxlZBIEYXR0biIHU29mdG1heCoUCgRheGlz"
        "GP///////////wGgAQI6AAoaCgRhdHRuCgJ2dBIEY3R4MCIGTWF0TXVsOgAKLAoEY3R4MBIEY3R4"
        "MSIJVHJhbnNwb3NlKhEKBHBlcm1AAEACQAFAA6ABBzoACiEKBGN0eDEKCE91dFNoYXBlEgRjdHgy"
        "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"
        "EgoQCAESDAoCCAEKAggECgIIEEIECgAQEQ=="
    ),
    "test_cpp_decomposed_gqa_wanda_pruning_matches_python_reference_exactly": (
        "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"
        "EBE="
    ),
    "test_cpp_decomposed_gqa_wanda_pruning_with_constant_mask_matches_python_reference_exactly": (
        "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"
        "BAoAEBE="
    ),
    "test_cpp_decomposed_mqa_wanda_pruning_matches_python_reference_exactly": (
        "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"
        "EBE="
    ),
    "test_cpp_dmmha_wanda_pruning_matches_python_reference": (
        "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"
        "EUIRCg1jb20ubWljcm9zb2Z0EAE="
    ),
    "test_cpp_gqa_wanda_packed_qkv_qk_norm_rope_pruning_matches_python_reference": (
        "CAo6iiUKGAoBWAoEV3FrdhIDcWt2IgZNYXRNdWw6AApBCgNxa3YKClNwbGl0U2l6ZXMSBXFfcmF3"
        "EgVrX3JhdxIBdiIFU3BsaXQqFAoEYXhpcxj///////////8BoAECOgAKKAoFcV9yYXcKDlFSZXNo"
        "YXBlMVNoYXBlEgRxX3IxIgdSZXNoYXBlOgAKXQoEcV9yMQoGUUdhbW1hEgRxX2xuIhxTaW1wbGlm"
        "aWVkTGF5ZXJOb3JtYWxpemF0aW9uKhQKBGF4aXMY////////////AaABAioRCgdlcHNpbG9uFb03"
        "hjWgAQE6AAorCgRxX2xuCg5RUmVzaGFwZTJTaGFwZRIIcV9ub3JtZWQiB1Jlc2hhcGU6AApfCghx"
        "X25vcm1lZAoGUG9zSWRzCghDb3NDYWNoZQoIU2luQ2FjaGUSBXFfcm90Ig9Sb3RhcnlFbWJlZGRp"
        "bmcqEAoJbnVtX2hlYWRzGACgAQI6DWNvbS5taWNyb3NvZnQKKAoFa19yYXcKDktSZXNoYXBlMVNo"
        "YXBlEgRrX3IxIgdSZXNoYXBlOgAKXQoEa19yMQoGS0dhbW1hEgRrX2xuIhxTaW1wbGlmaWVkTGF5"
        "ZXJOb3JtYWxpemF0aW9uKhQKBGF4aXMY////////////AaABAioRCgdlcHNpbG9uFb03hjWgAQE6"
        "AAorCgRrX2xuCg5LUmVzaGFwZTJTaGFwZRIIa19ub3JtZWQiB1Jlc2hhcGU6AApfCghrX25vcm1l"
        "ZAoGUG9zSWRzCghDb3NDYWNoZQoIU2luQ2FjaGUSBWtfcm90Ig9Sb3RhcnlFbWJlZGRpbmcqEAoJ"
        "bnVtX2hlYWRzGACgAQI6DWNvbS5taWNyb3NvZnQKgQEKBXFfcm90CgVrX3JvdAoBdgoACgAKCFNl"
        "cUxlbnNLCghUb3RhbFNlcRIDY3R4EgJwaxICcHYiE0dyb3VwUXVlcnlBdHRlbnRpb24qEAoJbnVt"
        "X2hlYWRzGASgAQIqEwoMa3ZfbnVtX2hlYWRzGAGgAQI6DWNvbS5taWNyb3NvZnQKGAoDY3R4CgRX"
        "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"
    ),
    "test_cpp_gqa_wanda_packed_qkv_split_pruning_matches_python_reference": (
        "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"
        "bWljcm9zb2Z0EAE="
    ),
    "test_cpp_gqa_wanda_pruning_cross_attention_matches_oracle_exactly": (
        "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"
        "ZnQQAQ=="
    ),
    "test_cpp_gqa_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference": (
        "CAo62xcKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKqQEKCkF0dG5CaWFzSW4KMEF0dG5CaWFzSW4vYXR0ZW50aW9uX2hlYWRf"
        "cHJ1bmluZ19nYXRoZXJfaW5kaWNlcxIqQXR0bkJpYXNJbi9hdHRlbnRpb25faGVhZF9wcnVuaW5n"
        "X2dhdGhlcmVkGihBdHRuQmlhc0luL2F0dGVudGlvbl9oZWFkX3BydW5pbmdfZ2F0aGVyIgZHYXRo"
        "ZXIqCwoEYXhpcxgBoAECCqsBCgFxCgFrCgF2CgAKAAoIU2VxTGVuc0sKCFRvdGFsU2VxCgAKAAoA"
        "CipBdHRuQmlhc0luL2F0dGVudGlvbl9oZWFkX3BydW5pbmdfZ2F0aGVyZWQSA2N0eBICcGsSAnB2"
        "IhNHcm91cFF1ZXJ5QXR0ZW50aW9uKhAKCW51bV9oZWFkcxgEoAECKhMKDGt2X251bV9oZWFkcxgB"
        "oAECOg1jb20ubWljcm9zb2Z0ChgKA2N0eAoEV291dBIBWSIGTWF0TXVsOgASAWcqjQgICAggEAFC"
        "AldxSoAIBtXlvwjl8L6pP7M8ESSFP3dKgr+gTs+9RvdQvgVJML7j8yW/4cOOPsjSWj6kHVo+G/EI"
        "PtgcLr6T5wrAznDdOyUypT8yrAzAiwfLP73Blr89wGC/+IqjP/E3CT71O+K+IDacv8eDLb9jWp4/"
        "Rm3QPthnjj7Siai/fvKUPpHXx79peAVA0wC6vldHib6230y//g3Wv9xMgD+2RtE9KSesPzefjD+C"
        "lr0+ZN01Ppj6A7+Tdj8/8tSVP3CEED+XP7e/tDPSP2eOfb9KzHA/EVg8PpI4271GCac/Sn/MPzdl"
        "jT5Wd9c/t4anPwPG7L5YB7A/Te2+PkLfwb92nby9wstFPw9CaD/Tb6y/G1pjvxVtEb+Qtpw/VJrY"
        "PdfOYL+lJxK+nF23P/9LHT5ZxqG/ovqGPy5IWr8VLYs/rMNRPn4pqj8orve+99xJPhoNsD7gSAJA"
        "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"
        "eL7zzag+iNKMPg2AvD+W0hpADVcFQNgVM7+li26/87p5v9G7Zr2qVN0/AKuNPs1Asj7zboC9Q1JB"
        "PjX0kz5KgXu+Am66PqA3079jCJw9VLDxP+CIJT+irCa/EgdwP6V7hr6Oq9C/0qZcP8CyBj+upAU+"
        "0E8Gv9noq78UXq0/QGFCv1pncz8Fci9A45nXvxGNqr/Ak3M/PumsPmGfhz9f5ZO/IuHTPw7WkT9i"
        "qKG//LcLv4U/Oj9P2ya/h9FZvwUk9r7Vb3o/QdqAP4BaWLxBx3K+kz0nQK8g5r7tEji/VAmzvbzS"
        "n782PsE7aBqUv4FXhT7oEv0/cLG8v7YzAj6CyAFAXZEovxxshL/0Jkm/kxdqv5LXyz7mpLI/DyqK"
        "vhxgFj9/eg9A22/TvffN3D+C86S9rkmaP8MyjLuvONM+Hg/PvXnOWz6qZRM/jTKJQE3mRD/777a9"
        "doezvqvExz/rawJAxhkNv2Qkbj4/hnO/HQ+6POGoKb6AEwG/ACEHP0wK3T7fe4Y9s9XPPOKMyb+o"
        "S+c/wrNHP/H/hz6IALg9fiHTP3cvpr4qGAgCEAZCCFNlcUxlbnNLSggEAAAABAAAACoSEAZCCFRv"
        "dGFsU2VxSgQFAAAAKlgIBBAHQjBBdHRuQmlhc0luL2F0dGVudGlvbl9oZWFkX3BydW5pbmdfZ2F0"
        "aGVyX2luZGljZXNKIAAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAMAAAAAAAAAWhcKAVgSEgoQCAES"
        "DAoCCAIKAggFCgIICFokCgpBdHRuQmlhc0luEhYKFAgBEhAKAggBCgIICAoCCAUKAggFYhcKAVkS"
        "EgoQCAESDAoCCAIKAggFCgIIBkIECgAQEUIRCg1jb20ubWljcm9zb2Z0EAE="
    ),
    "test_cpp_gqa_wanda_pruning_matches_python_reference": (
        "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"
    ),
    "test_cpp_gqa_wanda_pruning_per_head_attention_bias_matches_python_reference": (
        "CAo6txgKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKiQEKAXEKAWsKAXYKAAoACghTZXFMZW5zSwoIVG90YWxTZXEKAAoACgAK"
        "CEF0dG5CaWFzEgNjdHgSAnBrEgJwdiITR3JvdXBRdWVyeUF0dGVudGlvbioQCgludW1faGVhZHMY"
        "BKABAioTCgxrdl9udW1faGVhZHMYAaABAjoNY29tLm1pY3Jvc29mdAoYCgNjdHgKBFdvdXQSAVki"
        "Bk1hdE11bDoAEgFnKo0ICAgIIBABQgJXcUqACH+SIb/1220/F4EKPeRNN79eruA/Kupyvz2P1r4G"
        "Jrk//2pGv/wmGz76ghK/E/KWP/H5tT5WWV6/6/0dv4OQpD8wRqc90V2/P4T+rL8kR48/yjhwv+8h"
        "jb5SBVq+qEOtvkL0or/d97+8EnaiPq3GGz9EbKc/loqMv0nGBj+7zBm/vp+lv9s7v77s4uu/hi6v"
        "PkITsb496aG+xBMsPkqr7b/Gvtu/tK/NPiztOL8odWE/vtnwvllRyr97kYK/32Rwv0Yjsr0ucbi+"
        "9nkhvv1Mm7/QnPC+pzOjP8dlgT+d74K/b1y7vl8oXb+PNHw+38aavnXnb78As+496/Kqvz42cb7X"
        "VTXAqlKeP+7Zyzo0NbW/W/hXv5MOnD8S6j6/TObtvg1OV781xPw/718TPn/4sj8YWb0/zqw1vmoj"
        "I7+o2M2+O5GXv4B6ED912Is/Miz/P/P51b+oUyK/ZQh1PbzeC0APoAi+U0TyPsUB+r6GWEe/0VIS"
        "Pu/evTy/Pa8/IgifvyB2Lb/R4hTAYG8Tv3l3Dj4lggU/BmL2vohXhT/Y8r6/erNrP8s517/NpwLA"
        "f7ZCPtUXnT9thQC/6nsNQPVs0j3UZiC/dm/JPWd+lb2jZ0w/+UH/v1kOxLwWiUW//qwFv4RelT54"
        "RCg/4Z6zPuAvhr+ShPo+eKFbP5AFPL+SAlq9bQpBPjndiz9TIxk9wWBBQAmQEEBRDfo/M6KrP9Ge"
        "/D4qyaG9ZQi0P9hEjT/0OyC+cZM6Pu6e378HQlq+Pl2sP4fAYL/NQEC+tPn9PyZh7b9KBWU+r70o"
        "PRgpub6tntS+k7Wsv8Z64D4h13C+MyVSv7SJQT/ijI8/H8juPla1kj/Fktk/QjfLPvKXNr/MkHa/"
        "RUmbv66Mnr0nd04+YxCMv3LbR7+8glW/smTzv9RdbT+Rn7K+9f6yP4N/d7+ySj8/o7GTviG+xz7V"
        "Gew/DTeIPjkLpD+HUNC/RgDzv+ASBj/IlQE9ZJ+7PwRYhL8EL6U/FLZcvz/PfL+jaI2/erJ2v8AN"
        "qL+Mii0/zocHP1juFT7MxUy/ClVCvozbpL8v+G+/2TNJP2+3A8C03AbAyM8Kv6qICr4x8qW/0SvC"
        "Ph+ctb/xrWc/vPIkwOOFPr//rs69NnBcP90QtT+EVak+SsOWv/Gyir5t3Ai/R8iAPiw5ur7SYrk+"
        "38RdPwHHlT4IlyQ9YyqYv6LYQD8uK1S/dd5Rv7I/9b8Ywqo/uzVgvhAOST9zNme/+PvlPmfmhr9J"
        "76a8nwbIPer+mr67gUw+QpclwED7mz4UtW6/BtkSQLkhiz8yXp8/KXXGPkXfgT/yg6M+U27jv/7k"
        "XT6EIso/BSQfv8jbGr/EWdQ+j44sv15Unb8qjQIICAgIEAFCAldrSoAC6pKPvwZFTb5/o96+qZz0"
        "P76UOz/AHzM+DjqcPy1Agr/+zF+/NOsCvtK/Gz/R540/3cTwP2c3hL5zO22/3DtFPnKh4L9hCGO/"
        "H1j2PbpYV78nIJG/TlHJP4geTrme8Zu9ylviP3vntr98wLy+Ax2Tv/D3Gr97ht0+qHObP9vINEB6"
        "s3g+J/pNvwFh0T+ENuQ+A7bTviw6A8DgQxo/kVDJPxKyNL+/Hka/god0v34Z2T+EbZu8WvM+P2Gr"
        "6b+bK/G/37wsPxvE2zwbLkK/3ln8vvI27T21eWU+Iq3/v+pSKj8JphY/hva6Pr4Ylb/rKFo/M7xi"
        "vwpubT/IO1y+9vNxvyqNAggICAgQAUICV3ZKgAJ8qRO/6pDdvsO4jb9DM++/kL3cPr5C1r3Ci6i/"
        "Gkwxv8Ynk791XLM/T1gNv/CwMr7PD4O/3kyXvXMi/j/3A54/tOaqP5B5ur6f+vA7cbCpvziXjz6q"
        "shG8xD04v2Dogj5kJ3k+IgKLvzqv6L5L9YS+H6SaP9fpWb8+ZgW+A4EIvyIGjD440K4+kKTEPwle"
        "lr/oUfW/MjoXv5qw5b9zVpQ/gblTPo3tL78a1IY+uiByP5cS9z7kOao9yGqWv5Tkt74/PW4/REKW"
        "vqSv7L3ImQY9hHKRP0lpTL6bj6I8SyMrvtlg1T6kbOy/YkbYv8uJgD9ZPba/T2WpvpIcGL622bW/"
        "Ko8GCCAIBhABQgRXb3V0SoAGPY7mP24Qe78fSVC/ADrZv4JFib9L9h3AC/RWvk/0UT+EdIU/HBFs"
        "Psex2r6Jk1Q/Pz8MP3DcNT8MmgE+UTAOwPI64b69iHM9YxkTPyc1lL5oID2/n6OBv8atHL/3EeE+"
        "6VCgP4471z9theg+L/Q9P8UP3b6tUFA+bHn6P9VYnz+Tvq6+5BwFv4z4IL/nOmC/JSLSP8WJkz/U"
        "Nhi/XcDCvhndzL10GBK+6MD1PgHfp7/Wh8a97w8pvRyoyD8OTYU/cNQVvzS1+j08gx+9u9C6vomM"
        "Fj9+Vdc9BtUowIvV8D+AXC+9giXbvo+vl73jyj0/UNQ8v6G71L30FYw/anAHP+IWhD+k1Kg+uMUc"
        "P+yJu7+pjWk/OXIIPtThUL9oUyBAA8NQviZWIb9SuoU/XIHovonjrD+hnow9/bqIP25Jzb/E/Fw9"
        "6MUFPUZ9C724fqi9yKvKP72Crz5fhpE+cUIfPt3nWD/eJ6U+NrOtP54w7L5xENG89rbQv0jSZ791"
        "jJ68ds63v4V/dT8ZG/0+DlOGP2tWRb+SHQC/EO32PpOC1D203KU9SHS5P6t8Dz8jnbC+hgk+v3ym"
        "gz7xtUy/or+8vwiMgD+Y7h4+12XGvlA96z+F5bC/+hFkv5U7tL8Zbnk/1+59P+dA8r2aUJG/aRq3"
        "PmEtuz6lMpU/DOg0P0+SJD+sFga/zIqlvzwthj+ykQbAFYKSP0mFiD1JO5u/kbnsvUApqb9mDoM/"
        "ZfYAwNPlVr/5k3y/4YZWv6hxGz/9GCo/BbS5vo00kL6RFzY/xfO6v+oPxz/Ydd0+t48iQKokub5i"
        "J6S/Zbkzv9n+tD8Eo3E/EqOIveEfpz9WC3o+EncWP1WnDkDFmZE+eTn5vmigIT6GUZ6/UhedPmNe"
        "jz4kaKc/x7g7v1m+EL3mYLm+GtNXPxQQhj1RbvQ84M/cvhfUvD4uw3+/O161P2AHdb5IEJc+9Ov7"
        "v6yzLL9KOXY/mPcQvz4gRUCv9689gl5CP9PeYb9PyX0/q8jvvc/Nbb8THB6/KhgIAhAGQghTZXFM"
        "ZW5zS0oIBAAAAAQAAAAqEhAGQghUb3RhbFNlcUoEBQAAACqnAwgBCAQIBQgFEAFCCEF0dG5CaWFz"
        "SpADJZ92Pw+CQT9B7gW9Ww23PsF5eb/UI+w9anqJPnt3Yr9eAy687RNAvE5zN7/HHb6+EaA5PzQo"
        "Mb6BFGC9ZeiqP6tDAr4h6h8+8lhLv7H0OD+pEr6/pUzCv+m9gb+au9W/IdetP4EGML5Ok5Q/j7qh"
        "vyCNnr80QTW7l8vdO5sNfb/MRdW/rI6PvgM6Cj93pua9wcPkPjoOuj1mGo4+JYjuPhTlnj4/8HG/"
        "IkIvv6d2tj3Cxks/p3hNPi18d7xTu10/tU1tP+ZI2D3Esyy8LOblv7imPb9gLTE/YI4Lv/K+jj1H"
        "xsa/MC7fviwqEj++/jE/wMaLvozxBL2jXNe/1I4zPpOanL78PRu+0Dw2P2vKhz+TgRi/aAinP6hp"
        "jL8thCRARLRHvmp1Ir5edNU+C86dv+clOr9ixWa+gb0jv0yfPj8pVVM//bt1P80klT0SWsa/goUQ"
        "v2eW5D/J8Pq+hiHzvn1/Er9wk+M/c9yTvyx+Wb8LkUU/PjqGP6ayZL64al6+KWBKv97hvz4eMH+/"
        "UfghvloXCgFYEhIKEAgBEgwKAggCCgIIBQoCCAhiFwoBWRISChAIARIMCgIIAgoCCAUKAggGQgQK"
        "ABARQhEKDWNvbS5taWNyb3NvZnQQAQ=="
    ),
    "test_cpp_gqa_wanda_pruning_sliceable_past_kv_matches_python_reference": (
        "CAo6vRYKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKiQEKAXEKAWsKAXYKB1Bhc3RLZXkKCVBhc3RWYWx1ZQoIU2VxTGVuc0sK"
        "CFRvdGFsU2VxEgNjdHgSAnBrEgJwdiITR3JvdXBRdWVyeUF0dGVudGlvbioQCgludW1faGVhZHMY"
        "BKABAioTCgxrdl9udW1faGVhZHMYAaABAjoNY29tLm1pY3Jvc29mdAoYCgNjdHgKBFdvdXQSAVki"
        "Bk1hdE11bDoAEgFnKo0ICAgIIBABQgJXcUqACLC/tL79x+K/QLyOv85zvD/2wDg/K4LXvX9yAj8E"
        "Id6+fKkmvjn+Wj8TEj4/TWhdvlIUwz6dAJy+/i90vQraT7z6o7c/Y6lQvoT8cj91xuc+K9/wv7Rw"
        "pL0pxKW/AS7kPsl1sD/WP6Y/NwnLvzdf4T6sY8K+WM4/vyg8yT5324o+8bnFPgvmGb5Hsim/iXWC"
        "P7SZ+r76n+m/CrfAvWg4z76nGhy/dmeEP3QZkz9NiOa/4OWTP844DT/1eAg+namrPpa5JL6Tf9s9"
        "fPtPv7YwNj6L8ns+Wzahvxp6T79Ufw0/GYhvv38DCMB7LjA/LRBIP5OZDD+/gF8+IgqdP837kD9O"
        "4x+/2NP6vtlgA743DDm/ftqlPkWFyz4D/TS/rkoLwECrhr5cUx2+n4dEP5WyYr4eFTU+xxIoP4yw"
        "l7/J2mG/98tVP+yCUb58SlM/MpHlP0Dss7+Inus+NE95P13aTj659I6/KD9AP9iit719KAm/EykE"
        "vrODFTwHW5K8rA2Mv1koob8uJMs/1533vZwP8Tz9ORu+eloawGEGhj/FAIa/Ra3AvjT5lz+p3ew9"
        "MR5vPyjeDj9Ot7s/xJ0kvYt0xj9vG8k/09iRv0RCyz/Aa3+/QjXFvvvMQb97voI/RoWlPhxvpr33"
        "7KE/GLSnvp73N76dbwC/iRyWP4g0Uj+jOx8/kPerPxGTZr/8pI0+4p9DP3a35L9sGQY/HHEbP/ID"
        "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"
    ),
    "test_cpp_linear_attention_wanda_pruning_matches_python_reference": (
        "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"
    ),
    "test_cpp_mha_wanda_pruning_matches_python_reference": (
        "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"
        "b3NvZnQQAQ=="
    ),
    "test_cpp_onnx_attention_wanda_packed_qkv_native_rotary_embedding_pruning_matches_python_reference": (
        "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"
        "ChAIARIMCgIIAgoCCAUKAggGQgQKABAY"
    ),
    "test_cpp_onnx_attention_wanda_pruning_diff_v_head_size_matches_oracle_exactly": (
        "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"
        "CAVCBAoAEBg="
    ),
    "test_cpp_packed_mha_wanda_pruning_empty_calibration_data_matches_python_reference": (
        "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"
    ),
    "test_cpp_paged_attention_wanda_pruning_empty_calibration_data_matches_python_reference": (
        "CAo6lBoKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKrgEKAXEKAWsKAXYKCEtleUNhY2hlCgpWYWx1ZUNhY2hlCglDdW1TZXFM"
        "ZW4KC1Bhc3RTZXFMZW5zCgpCbG9ja1RhYmxlEgNjdHgSDWtleV9jYWNoZV9vdXQSD3ZhbHVlX2Nh"
        "Y2hlX291dCIOUGFnZWRBdHRlbnRpb24qEAoJbnVtX2hlYWRzGASgAQIqEwoMa3ZfbnVtX2hlYWRz"
        "GAKgAQI6DWNvbS5taWNyb3NvZnQKGAoDY3R4CgRXb3V0EgFZIgZNYXRNdWw6ABIBZyqNCAgICCAQ"
        "AUICV3FKgAi8Hkw946XivWAdh74ilBY/e49LvkmCyr4y2I+/HHZKPv3YhL7a9Sq+9oucPzkwpD5f"
        "5zA/+Qt6PegCJj9nl1u/jifPPo9h1rxqc4C9BcYoPPMo8745Bbq+MzavPeZatj7AuQy+yWamvnF8"
        "C796aEg9UxEAv+bRCL1I7xs/42QOv9IIcb/Y1sE9Okv3vb8gCb/7iOQ9w7Iev3sNSb45mT49EYR7"
        "vdenD78DI/E+le8Jvlk0ij6PaSk/h0CLP7t8kD//IGA9qN4APyGMHL7D3oy9GSQNPrQd4b4hGLw/"
        "ZGz0Pq58/T41WUg/L/FlPgy3Nr9Mo0m/cJj1PYuA9D7Ji4q96AkMvsDeJb4Mobk+zqA9vjDPMD/B"
        "fMy9sHlEvoZGHT9KUR8/SfPjvurfJz8X/Mw+F4w9v4kKvz6V6h+/eL8FPwGbvj50ZE+/iOs/v+YJ"
        "Ij9dCJm/1Huxvid5Cj7kA2k/VDcpP6gDyj7yLnU/8QlyPRmpPz6DH7q+fTX5Pvp9bz4D/F2/P5Sq"
        "vpxHMj92NZC+FlHiPuaEe76ZHpk+B99kv969VT5TlJ0818n1vlwfYDsV68m+r+CmPkMjNj/kY+m+"
        "dSnIvjwEab7+/d6+4AHuvBI/mb9lEPM+5/ERP+YqUD+qxM++pF3qPrOb6T6K68a+IA2DPunXVD3h"
        "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"
        "Y29tLm1pY3Jvc29mdBAB"
    ),
    "test_cpp_sparse_attention_wanda_pruning_empty_calibration_data_matches_python_reference": (
        "CAo6wRoKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKxQEKAXEKAWsKAXYKB1Bhc3RLZXkKCVBhc3RWYWx1ZQoGUm93SWR4CgZD"
        "b2xJZHgKCFRvdGFsU2VxCgtLZXlUb3RhbFNlcRIEYXR0bhIKUHJlc2VudEtleRIMUHJlc2VudFZh"
        "bHVlIg9TcGFyc2VBdHRlbnRpb24qEAoJbnVtX2hlYWRzGASgAQIqEwoMa3ZfbnVtX2hlYWRzGAKg"
        "AQIqGAoRc3BhcnNlX2Jsb2NrX3NpemUYEKABAjoNY29tLm1pY3Jvc29mdAoZCgRhdHRuCgRXb3V0"
        "EgFZIgZNYXRNdWw6ABIBZyqNCAgICCAQAUICV3FKgAg/2Le9z29Mvbf4rr1zV1S+StfkPKF4vj1D"
        "CF49mxGFvbTihj3nEwy+In3cvIXgTjyDp5W8DThrvUNxnL3xmBO9i06IPRkBAD1Dc1G9sjCAPR1P"
        "x73gYDO+FZkRvgbtk7uqmMy87c2tvWtUQbwSZ1e+wyNePXraQL1dwle+vR+EPU3xdT73OWq8+Wmq"
        "OxuHgL68RKW7SXNDPb7KO76oVO4710gPvrMEJT0TQoS9daptPqa1AL1Do8K9XKEKPfM9Pz0N3om9"
        "SwWVPRxRor1tOzi9JT8Ivelty7wVFDs97X5zPdqNgT5MNko7B7YzvCCTUD2PbKq9TfMKPlsfST0Q"
        "LzM9H3XNPck2Ab4MYJo9aTMbvova9r0l3fY9XytEugr64LyWSDO+ewCQvUiETTy3tz88d6+3varb"
        "hb0gZdc9oswzPcde3723AZU7GA8tvojafr0fs968S56YvYxGCb7P9VM9ptIsvmaaxLyizeU9n0UO"
        "vb/4H71Bl5k9b3woPjctB71uG0e+FxD9PWhA5b039yq+nSMVPJNjQT2vzm+9aFipvgdgzr0oj8U8"
        "RBYAvOPLHj2bXiq+aLiAPmAUubsdux69w4QBvvatwzwjP2u+R7QoPpO31z2a+yi+d8k/vGNd/b2S"
        "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"
        "dBAB"
    ),
    "test_cpp_sparse_attention_wanda_pruning_nonempty_past_kv_constant_is_sliced_matches_python": (
        "CAo6rR4KFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKxQEKAXEKAWsKAXYKB1Bhc3RLZXkKCVBhc3RWYWx1ZQoGUm93SWR4CgZD"
        "b2xJZHgKCFRvdGFsU2VxCgtLZXlUb3RhbFNlcRIEYXR0bhIKUHJlc2VudEtleRIMUHJlc2VudFZh"
        "bHVlIg9TcGFyc2VBdHRlbnRpb24qEAoJbnVtX2hlYWRzGASgAQIqEwoMa3ZfbnVtX2hlYWRzGAGg"
        "AQIqGAoRc3BhcnNlX2Jsb2NrX3NpemUYEKABAjoNY29tLm1pY3Jvc29mdAoZCgRhdHRuCgRXb3V0"
        "EgFZIgZNYXRNdWw6ABIBZyqNCAgICCAQAUICV3FKgAjzwBe8VQhnPmGvvTsPvYs+uMH2PUsAjD29"
        "kPE9t8maPr57LzwomhA9zhQ0PYu56727vEE9ldO3vU0sL71yC2k+C9smPjvQWb0FRNM9kXqJPTu9"
        "Eb49XHA+Dn3HO8S/wj1jMEs9Jwg0PYddqb3R4oK9qF5/OzdkeD7oTXA9E13ZPY4tpL2kdwQ9j3AH"
        "vArO7by+iTC8P0VPvp49rjx4Wew9+rp0ul3DVT3w5os92ucIvrMJmT0ihMo9CdOLvCG9Sz3vz9K5"
        "Rjk2vjdGlz0LvVM9LQhnvdViu70L8M27ICD3u8VTM77bswi+RBwCvacXlT2b92e+XugDvSLFXb06"
        "hP09tZhwPiA93T0CMts9m8SIPjvTD77r4ve98wK4PFfMcD01CI49gkKkPVskuzxxySg9DcRuPQfN"
        "BD2/AuU9ul/HPSV6Uj2iFQu+a4QXvgnEmDxRDcW8JLIhvpJnVT3eYKA9f4nQPJ4Kr72RUJo9kEeb"
        "vUs2LD5C/hs+n3lrPXtbw7006is9pq+rPXHmv71+B7K8rR92vUeEFr6dV7y9p8aGvXUTPT5KSw6+"
        "r4BZPn3Raz2VNeq9mBjxPXu0eDzedUY953LZPZ1h2j3/4gK+FYLzvaUJVr14Jc49U2NuuxnOiDuT"
        "F829bGiFvbDQ+j1XdDi9ag4MPSGOpD19+gS+fKW8urd/DD0nJ2Y7YJTKvMiLxr2CEj89ADFYvN0Z"
        "Fz5ihlQ9NblhPaLyiztH38s9jC8YveLLhjyfqtg8l9PwPe8lUDyz23M9LCW7vbaJNL499L+9kltO"
        "O337dj2XFmu8EIhavhzJNj1a/pm6APalvZXT+Dz7+k09ByyyPQaVED5QKMs9l1fSva9bJT0zOhc+"
        "yAmzPIYctz1TTJq9Aw0YvqWH/TzrLTA9N9e4PJujjD3v7569uDSmO6Bl3D1YFGc9bxDhvc/UOD7U"
        "ch+9nTiVPWAKOjxa/tO9+4cRPlTQmj1dZME8hxU1PXip7T0FnYg8o8XpveJbsDw9Z1W+EEiCPU2D"
        "7j3n5AU+UNV4vMUF5bwDlUm9S/XdvB2SEL2BxkW9E5aOvnNNhL2qUtq9vcECvvcRi739l/c9YrgS"
        "viIwpD1Du1+9GHHUPe4/Ir6ykWo9mXgXvQsCXr0Bj8u7GlPdPHJBnL0odM49cOfAvXS4Sj0vzzE8"
        "WMeFPd9gJr7QdXG8sFf+PCOWmz0d/ki+4ausPN/MI74uQic+uB/tvHfdJbwOBLA9s94uvijNMj7b"
        "Auu9sB8WPqeQej79GaI9s1p/PpKsc72nFVc95VJoPuL+Jj3ru3U8RwgWPoLOOLtBrDs9kW2tPJBT"
        "0zyY6Py971lfPYBCqz36ddi9MgVSPpNPjD09nGE9Ko0CCAgICBABQgJXa0qAAqH2uT2tYfg9PDyO"
        "PH4UKb4XorU9of4wvUIdMbwDP6q9FzmUvb8Stjxtuk+9AubpPF1Igj70giK9tdmwvK3JzT0nUvS6"
        "h7DsPP/lWjptTBC+LX5OPWgV+b33MAE9tHLIPcsJ1L2zHfW92gvnPMW/k7zvLfs8BZOPvNuXP70j"
        "zai8ObwYvSLMQ7y9hOM6qzz1PfM5Zjs3MSO+jps3vim6Nz6lyTo+nS3avTc0Fz4jkgs9R6CsvZiE"
        "77038pG90hmVu238mDxvHsq6jZzUPSdK37xTZ8K9bs9JvS2ouz2Ia+08gFU8PYXwk72+tQm+QqVj"
        "vWsDoD2lifw7GvpwvZ4VjL0qjQIICAgIEAFCAld2SoACBxstPFPw+z342Bi+DwYRvtWvBL4egJS8"
        "YkzvOsYotj2r1po99u+LPmcOpr16tFM98HtlPQt/vD1STGu+CHvFPbCVCz0tsMQ7B6GiPW9tej26"
        "1le9nZGxvX/jID6MSEe8e04UPmd9Pj2fX+C942myPYvWrDzo6eG7MtC+vbKCYr0IOEU+q6/WvMye"
        "iD1ofLs97lu7vKJIz71FfUy9UnEpPrKCEj2VjqM8F7NdvSUlBT5jF3K9r3UQPr1OOz3BbyU+WLXx"
        "vKeyQb2T5Zw7AuDovWAoszyyu947FCDEvUedlj1FrYO91ftRvUImQr1A0QK+av0yPLUiDL5copc8"
        "3PsVPSqPBgggCAYQAUIEV291dEqABo/iGD3irP093bTPvSvWHT1awzw9MiznvEgh6D0E6qq9YQVB"
        "vkI9Sj2XUns90qaXPCIVF70WfKS9nVX9PTDjgD2MzAG+bSngPbgfjb0MtqQ906EHvRDeTr5DaYy9"
        "F7vdvcP1HT7RJbO9f2Y+PYeCqT3lSoS9060YPtyClD7qqws94tu9PdKT3711eBy8h/Kaven6Qr0d"
        "JdQ9mxmyPELN6Tw8eg6+QYSBPe0zGz1vQOY8B5h/PptKvT3kwSS+h1GwvdY1Kb7v6gm920JsvT7j"
        "Pj5DQVa8Z/T1vQNiCTvIxGo9nFEhPo1hy7wVhf69sNvkvXAr3j2G+wU9zGGAvUDPLb1rZ449Wrsz"
        "vp4BNryYFHY9EFuoPbFDxLw4Few9AnwlPvsikz0J/BY9svMGvaPQCz7uAaE8+FOQPVCZ9LyGTJk6"
        "vvk+PtskWT2fqA2+JxtIvPxbiT0qgjY+ij1sPSn9JT05vEw9PVORvkPjmb2ABhM9yfnGPDTknr0v"
        "XRi+Ox8OvcpgDz6oPcs8K2GZPTjcU73i4c+8N9SPvR+kc70xKyQ9aze6vG8brL3qQpq9T0QEPtVN"
        "Z716Du+8irF7PeKzXb1M7qO9JyIhvlC4kLw9sa09Yu8Vu5gsiL3s+oA9oHdFPEYmLL31SL+9+0gw"
        "O9Xn2jzooVk9xQJvvqqnfL2e8aQ9Lz4hvfQZgTy663y9UdadPc/5TT2L/7K9ZYgsPZRuFz6cxIy9"
        "H/Vsvq4Mkzxufr88ldE6vivwvL2Tqnw+Z7UlvL18AT7z/Jo9cuH8vRiSNTwwPhq9O+BIPOeYE77m"
        "VBg9WWIAPlg65rybXtS9sSERvupRKb2Lwky9JAKyvWPiob04VnI+tTBovcSRObwaXUo+fS5sPlgs"
        "k73qB/G9F9dRPVBJLL1618C9RlHCvShUcjwzrdo9dnKEPtP4djww3xY+7+FLvv9YXr2XSCs8lS2M"
        "vHCVYb3Cm/69D7lqu4HsKb7i7d29baFwPTQ3Gj4dWli92ATaPKdwar5SZfQ97xLAvSoYCAEIAhAG"
        "QgZSb3dJZHhKCAAAAAABAAAAKhQIAQgBEAZCBkNvbElkeEoEAAAAACoSEAZCCFRvdGFsU2VxSgQQ"
        "AAAAKhcIARAGQgtLZXlUb3RhbFNlcUoEEAAAACqWBAgBCAEIEAgIEAFCB1Bhc3RLZXlKgARLPX+9"
        "cAw9vtta1z2jzK29GLQJPFDATT5j6u+9ytbWPWgTYD7X+Bm9n5GePX9puT3MP0s6k58AvHMZIL00"
        "fjU+UxBvvsPBkbwoUY29/IwbPYRXtb0ZXo87hIBMPB45mz2/f7e9tew5vSWySbwqM5e9fP2NvZcx"
        "773/iju9sIASPiPAs72ajHk9rUanvaJgdj1PS4Y9l3QmPu3GXD2ojYw9X/HSPaN09L22JI67GMTz"
        "vHgAL7wI/I683wRoOu4ipD2OeKw91dNyPXLhnT1jTLY81VEhPojYCD6vKg28IkuhPbPjWD0ZK5O9"
        "ooPXvMBDZD2Tgza8iJJTPfdgvr1nCTi9mwqUPhubUL6rSQc9SzSyvksHNj0h54a89e0BPg2GQj4t"
        "b188idUbPnMeqbxC6D48ENLRPPOBPD0YnGa8VV7YvROtMb2fhLe8AzS2vMr4pzy/DjG9p9hrPdWs"
        "5LyrQoo9egU1vcUMxr23E6q9g1c7PttApj0d3oo9KxSxvRbTljx7msq9MesFvbIB4z14gQ49FTgz"
        "Povhsb0rkhU+OIqYvWvJ4LzFqGw9X7ytvTJ/ejyfg809UqSbPbdAT7xPrjI8PVxzOqc9Mz33ydU8"
        "3zs6vUPn9TxvoBc+E0rivPuXZr2T9S89E4t7PaIRIj1tLe08GUC3vXXgs72rOoU+VyUnPSqYBAgB"
        "CAEIEAgIEAFCCVBhc3RWYWx1ZUqABFXG0rw5EJi9c6/IvWMuqT3HOtm8hQZzvaKiiT7iGQS+/tkV"
        "vorBLzw1Gyq+2L16vfbWF75dAk07hjw2vq/kqzw/vki9jWG0PWrjrzxVl2g9sVQqPfBV77zBzca9"
        "I4EXPABbrbwdGKa9ZdkHPgMrIb7hWxK+5c/mvHujSD3o6O68B+vlOYtHAT44Wqa9YO1dvgKQoj1z"
        "LNS7D/EyPcuOYz2C2/87r+f8vcI0Xr6cvbc8Y9YBProOrT1JNbm8d7Mvvgu49T3qzg2+mtGNPZt9"
        "Kj23OWW6PYpau9KqOT01kc+9Y/FhPR/GQD5Rs4k+IIgwvAvFdb1N+fM8+Pr0PTvK1rxGDEC+TaIF"
        "PksA8D2HI9o8IrUZvogJCb7oeIU8J6ZMPH6DlD4g9qM8rgMoPpAkPL0v/k67ILVqu/S6pjyeQDu+"
        "r/BVvRa1qb3P8qc9rIKGPOlLkL2L1Ni9OrgavRu0NL23JaW8JaW+PVPQF71eoqK9EQfEPGJQQL7D"
        "/cw8poGLPa1W9D0V9nG83+HzPftcqDw6Sjg9cgyLPkuAWL7v3so9L9XePbO1pr2T2p07hxbXu49C"
        "C77Yf+C9+K1cPjOtFj2braQ9PYGePFCUMT0rzms96N83vROKkr1eVZU94vlYPnhThD1CJNi9UVYi"
        "Pi9zCD7fHSO+34Z0vsv5rD4aoic+WhcKAVgSEgoQCAESDAoCCAEKAggQCgIICGIXCgFZEhIKEAgB"
        "EgwKAggBCgIIEAoCCAZCBAoAEBJCEQoNY29tLm1pY3Jvc29mdBAB"
    ),
}

_GOLDEN[
    "test_cpp_attention_head_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2"
] = (
    (
        "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"
        "AQ=="
    ),
    (
        "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"
        "AQ=="
    ),
)
_GOLDEN[
    "test_cpp_gqa_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2"
] = (
    (
        "CAo6vAwKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKeQoBcQoBawoBdgoACgAKCFNlcUxlbnNLCghUb3RhbFNlcRIDY3R4EgJw"
        "axICcHYiE0dyb3VwUXVlcnlBdHRlbnRpb24qEAoJbnVtX2hlYWRzGAKgAQIqEwoMa3ZfbnVtX2hl"
        "YWRzGAGgAQI6DWNvbS5taWNyb3NvZnQKGAoDY3R4CgRXb3V0EgFZIgZNYXRNdWw6ABIBZyqNBAgI"
        "CBAQAUICV3FKgAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAACqNAggICAgQAUICV2tKgAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKo0C"
        "CAgICBABQgJXdkqAAgAAgEEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqzwEIEAgDEAFCBFdvdXRK"
        "wAGFjoa/a1a5vrfGTz9VKKC/mVCjv5txiT+rD5c/NinfP/EdNb4MoHI//R4JP/aCcz5ysE8/P7ia"
        "PpEm5j8TqDQ/vgb3vnqvUT92B3A/XZTnvzsEqj5gqnK8WwiCOyWCj7+te6C/69KnP6vatj/AisU/"
        "dfo9v+cedr+P52++TAUMP3z+cz+XFQc/tjzBPmqT9LvlQPm+pthTP3oErr8p2tg/6wD0vpXo+b80"
        "fNs+lFiQPr3hhD9oEVw/+e2HvgWqbz8qGAgCEAZCCFNlcUxlbnNLSggEAAAABAAAACoSEAZCCFRv"
        "dGFsU2VxSgQFAAAAWhcKAVgSEgoQCAESDAoCCAIKAggFCgIICGIXCgFZEhIKEAgBEgwKAggCCgII"
        "BQoCCANCBAoAEBFCEQoNY29tLm1pY3Jvc29mdBAB"
    ),
    (
        "CAo6vAwKFAoBWAoCV3ESAXEiBk1hdE11bDoAChQKAVgKAldrEgFrIgZNYXRNdWw6AAoUCgFYCgJX"
        "dhIBdiIGTWF0TXVsOgAKeQoBcQoBawoBdgoACgAKCFNlcUxlbnNLCghUb3RhbFNlcRIDY3R4EgJw"
        "axICcHYiE0dyb3VwUXVlcnlBdHRlbnRpb24qEAoJbnVtX2hlYWRzGAKgAQIqEwoMa3ZfbnVtX2hl"
        "YWRzGAGgAQI6DWNvbS5taWNyb3NvZnQKGAoDY3R4CgRXb3V0EgFZIgZNYXRNdWw6ABIBZyqNBAgI"
        "CBAQAUICV3FKgAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAACqNAggICAgQAUICV2tKgAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKo0C"
        "CAgICBABQgJXdkqAAgAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8qzwEIEAgDEAFCBFdvdXRK"
        "wAEUlR2/yc2FPtuzt78oRTC/XA0lvj4Chb+pFZI/krcmQKN3l778r/i+x9fWv7t70T14o6S+UiUM"
        "PXGVo7+6Pwy/6tCdvllswj+IXYe/aQnsPmp6lz7Q0L6+SmVtv/2INT+JDbi+6YGfP6B25T9CubC/"
        "RC8hPm0T+b67ZMi/8aKPv7nkFj9oUs4/RLSTPrLOoT9h0hY/mKSfPxsiY7/5tpy/2zcSPwkLtL+S"
        "Jry/5xLuviaSPz+osei+ohEaPtrRM78qGAgCEAZCCFNlcUxlbnNLSggEAAAABAAAACoSEAZCCFRv"
        "dGFsU2VxSgQFAAAAWhcKAVgSEgoQCAESDAoCCAIKAggFCgIICGIXCgFZEhIKEAgBEgwKAggCCgII"
        "BQoCCANCBAoAEBFCEQoNY29tLm1pY3Jvc29mdBAB"
    ),
)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


def _probe_act_norm(model, probe_name, feeds):
    # Mirrors bias_correction.py's own `_add_probe_outputs`/pruning.py's own
    # `_wanda_attention_calibration_stats`: expose `probe_name` as an extra
    # graph output, run the (unmodified otherwise) graph, and reduce over
    # every axis but the last (channel) one -- the same reduction
    # WandaCalibrationStats performs in structured_pruning_entry.cpp.
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=probe_name))
    _, act = _run(probe_model, feeds)
    act = np.asarray(act, dtype=np.float64)
    reduce_axes = tuple(range(act.ndim - 1))
    return np.sqrt(np.mean(np.square(act), axis=reduce_axes))


# --- Plain com.microsoft::Attention (merged QKV weight) ---------------------


def _attention_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=True,
    with_reshape=False,
    wqkv=None,
    bqkv=None,
    wout=None,
    num_heads=None,
    attention_bias=None,  # constant/declared-shape attention_bias array, or None
    attention_bias_dynamic=False,  # declare AttentionBias as a graph INPUT instead
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + Nk + Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    if bias and bqkv is None:
        bqkv = rng.standard_normal((Nq + Nk + Nv,)).astype(np.float32)
    heads = H if num_heads is None else num_heads

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    extra_inputs = ""
    operands = ["X", "Wqkv"]
    if bias:
        initializer.append(_f32(bqkv, "Bqkv"))
        operands.append("Bqkv")
    else:
        operands.append("")

    # `attention_bias` (index 5) sits behind `mask_index` (3) and `past` (4),
    # both always left unconnected here -- threaded through as empty
    # positional placeholders to reach index 5.
    if attention_bias is not None:
        operands += ["", ""]
        if attention_bias_dynamic:
            shape_str = ",".join(str(d) for d in np.asarray(attention_bias).shape)
            extra_inputs += f", float[{shape_str}] AttentionBias"
        else:
            initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
        operands.append("AttentionBias")

    while operands and operands[-1] == "":
        operands.pop()
    qkv_inputs = ", ".join(operands)

    if with_reshape:
        shape = np.array([batch, seq, Nv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[batch,seq,{K}] X{extra_inputs}) => (float[batch,seq,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> ({qkv_inputs})
          {tail}
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K, H=H, D=D, Out=Out, Nq=Nq, Nk=Nk, Nv=Nv, wqkv=wqkv, bqkv=bqkv, wout=wout
    )


def _attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "Attention")


def _attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    return num_heads, qkv


def _plain_attention_importance(wqkv, nq, nk, nv, num_heads):
    dq, dk, dv = nq // num_heads, nk // num_heads, nv // num_heads
    wq, wk, wv = wqkv[:, :nq], wqkv[:, nq : nq + nk], wqkv[:, nq + nk :]
    importance = np.zeros(num_heads)
    for h in range(num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = np.linalg.norm(block)
    return importance


def _wanda_attention_keep_heads(wqkv, nq, nk, nv, num_heads, act_norm, keep_count):
    dv = nv // num_heads
    base = _plain_attention_importance(wqkv, nq, nk, nv, num_heads)
    act_head = np.array(
        [np.linalg.norm(act_norm[h * dv : (h + 1) * dv]) for h in range(num_heads)]
    )
    importance = base * np.maximum(act_head, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_attention_head_wanda_pruning_matches_oracle_and_differs_from_plain():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((3, 6, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep = _wanda_attention_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], act_norm, 2
    )
    plain_importance = _plain_attention_importance(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"]
    )
    plain_keep = np.sort(np.argsort(-plain_importance)[:2])
    assert not np.array_equal(keep, plain_keep)  # calibration actually matters here

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2

    d = cfg["D"]
    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=8,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
    )

    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    # Confirm plain (magnitude-only) pruning on the same weights would have
    # kept a genuinely different -- and, against this calibration signal,
    # worse -- head set.
    plain_pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    (y_plain,) = _run(plain_pruned, {"X": x})
    assert not np.allclose(y_plain, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_wanda_pruning_reshape_hop_matches_oracle():
    model, cfg = _attention_model(
        K=8, H=4, D=4, Out=6, seed=12, with_reshape=True, batch=2, seq=5
    )
    rng = np.random.default_rng(13)
    x_cal = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    # The probed activation is the output projection's OWN input -- here
    # that's the Reshape's output ("ctx2"), not the Attention node's raw
    # output ("ctx") -- exercises `chain.consumer_node->input(0)` resolving
    # past the Reshape hop exactly as pruning.py's own
    # `chain.consumer_node.input[0]` does.
    act_norm = _probe_act_norm(model, "ctx2", {"X": x_cal})
    keep = _wanda_attention_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], act_norm, 2
    )

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Attention", "Reshape", "MatMul"]

    d = cfg["D"]
    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=12,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
        with_reshape=True,
        batch=2,
        seq=5,
    )

    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_wanda_pruning_empty_calibration_data_matches_plain():
    # An empty (but present) calibration_data means no activation was ever
    # observed for any probe point, so every matched block falls back to
    # apply_attention_head_pruning_cpp's own plain ||W||_F ranking -- exactly
    # byte-identical output.
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=10)
    wanda_empty = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


# The plain com.microsoft::Attention family's own optional `attention_bias`
# input (index 5) now reuses the exact same HeadBiasInputIsSafe/
# SliceOrGatherHeadBias machinery whether reached from the plain
# (`apply_attention_head_pruning_cpp`, see
# ``test_attention_head_pruning_cpp.py``'s own dedicated coverage) or this
# Wanda-calibrated entry point -- both dispatch through the identical
# ApplyOnePlainAttentionChain, just with a real calibrated `act_norm` map
# threaded through here instead of `nullptr`.
def test_cpp_attention_head_wanda_pruning_per_head_attention_bias_matches_python_reference():
    H, D, seq = 4, 4, 5
    bias = (
        np.random.default_rng(124).standard_normal((1, H, seq, seq)).astype(np.float32)
    )
    model, cfg = _attention_model(K=8, H=H, D=D, Out=6, seed=124, attention_bias=bias)
    rng_cal = np.random.default_rng(125)
    x_cal = rng_cal.standard_normal((2, seq, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_attention_head_wanda_pruning_per_head_attention_bias_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()


def test_cpp_attention_head_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference():
    H, D, seq = 4, 4, 5
    model, cfg = _attention_model(
        K=8,
        H=H,
        D=D,
        Out=6,
        seed=126,
        attention_bias=np.zeros((1, H, seq, seq), dtype=np.float32),
        attention_bias_dynamic=True,
    )
    rng_cal = np.random.default_rng(127)
    x_cal = rng_cal.standard_normal((2, seq, cfg["K"])).astype(np.float32)
    attn_bias_cal = rng_cal.standard_normal((1, H, seq, seq)).astype(np.float32)
    calibration_data = [{"X": x_cal, "AttentionBias": attn_bias_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_attention_head_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    assert gather_nodes[0].input[0] == "AttentionBias"


# --- com.microsoft::GroupQueryAttention (separate Q/K/V producers) ---------


def _gqa_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nkv,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nkv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    operands = ["q", "k", "v", "", "", "SeqLensK", "TotalSeq"]

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _gqa_node(model):
    return next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")


def _gqa_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size):
    group_size = num_heads // kv_num_heads
    importance = np.zeros(kv_num_heads)
    for kv in range(kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * head_size : (h + 1) * head_size]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * head_size : (kv + 1) * head_size]
        v_block = wv[:, kv * head_size : (kv + 1) * head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return importance


def _wanda_gqa_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, act_norm, keep_count
):
    group_size = num_heads // kv_num_heads
    base = _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size)
    act_group = np.array(
        [
            np.linalg.norm(
                act_norm[
                    kv * group_size * head_size : (kv + 1) * group_size * head_size
                ]
            )
            for kv in range(kv_num_heads)
        ]
    )
    importance = base * np.maximum(act_group, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_gqa_wanda_pruning_matches_oracle_exactly():
    # Calibration and eval data must share the model's own fixed batch/seq
    # (seqlens_k/total_sequence_length are baked-in constants tied to a
    # specific batch/seq -- see _gqa_model -- a real GroupQueryAttention
    # KV-cache-bookkeeping constraint, not a limitation of this pass).
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep_groups = _wanda_gqa_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], act_norm, 1
    )
    group_size = cfg["H"] // cfg["KVH"]
    keep_q_heads = _group_q_heads(keep_groups, group_size)

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == len(keep_groups)
    assert num_heads == len(keep_q_heads)

    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=8,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_gqa_wanda_pruning_empty_calibration_data_matches_plain():
    model, _ = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=10)
    wanda_empty = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


# GroupQueryAttention's own optional attention_bias/past_key/past_value inputs
# now reuse the exact same HeadBiasInputIsSafe/SliceOrGatherHeadBias/
# PastKvConstantsAreSliceable/SliceKvCacheAxis1 machinery whether reached from
# the plain (`apply_attention_head_pruning_cpp`, see
# ``test_attention_head_pruning_cpp.py``'s own dedicated coverage) or this
# Wanda-calibrated entry point -- both dispatch through the identical
# ApplyOneGqaChain, just with a real calibrated `act_norm` map threaded
# through here instead of `nullptr`. A small, local `_gqa_model_ext` (rather
# than widening this file's own shared `_gqa_model`, used by many pre-existing
# tests above with a fixed operand list) covers just the two new roles this
# fix closes, to confirm the fix also holds up end to end through THIS entry
# point's own separate graph/used_names/value_info_by_name plumbing
# (ApplyAttentionHeadWandaPruning -> ApplyAttentionChains -> ApplyOneGqaChain).
def _gqa_model_ext(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    attention_bias=None,
    past_kv=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    wq = rng.standard_normal((K, Nq)).astype(np.float32)
    wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nkv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""
    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    else:
        operands += ["", ""]
    operands += ["SeqLensK", "TotalSeq", "", "", ""]

    if attention_bias == "per_head":
        bias_t = rng.standard_normal((1, H, seq, seq)).astype(np.float32)
        initializer.append(_f32(bias_t, "AttnBias"))
        operands.append("AttnBias")
    elif attention_bias == "dynamic_per_head":
        operands.append("AttnBiasIn")
        extra_graph_inputs += f", float[1,{H},{seq},{seq}] AttnBiasIn"
    else:
        operands.append("")

    while operands and operands[-1] == "":
        operands.pop()

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(K=K, H=H, KVH=KVH, D=D, Out=Out, batch=batch, seq=seq)


def test_cpp_gqa_wanda_pruning_per_head_attention_bias_matches_python_reference():
    model, cfg = _gqa_model_ext(seed=120, attention_bias="per_head")
    rng_cal = np.random.default_rng(121)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_gqa_wanda_pruning_per_head_attention_bias_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_gqa_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference():
    model, cfg = _gqa_model_ext(seed=122, attention_bias="dynamic_per_head")
    rng_cal = np.random.default_rng(123)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    attn_bias_cal = rng_cal.standard_normal(
        (1, cfg["H"], cfg["seq"], cfg["seq"])
    ).astype(np.float32)
    calibration_data = [{"X": x_cal, "AttnBiasIn": attn_bias_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_gqa_wanda_pruning_dynamic_attention_bias_gather_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert any(n.op_type == "Gather" for n in pruned_cpp.graph.node)


def test_cpp_gqa_wanda_pruning_sliceable_past_kv_matches_python_reference():
    model, cfg = _gqa_model_ext(seed=124, past_kv="nonempty")
    rng_cal = np.random.default_rng(125)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_gqa_wanda_pruning_sliceable_past_kv_matches_python_reference"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["PastKey"].shape == (cfg["batch"], kv_num_heads, 1, cfg["D"])


def _gqa_cross_model(
    K_dec=8,
    K_enc=6,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    # Q fed from a different producer/input (`Xdec`) than K/V (`Xenc`) -- a
    # real, valid shape (e.g. encoder-decoder cross-attention): Q's own
    # producer weight has `K_dec` rows, K's/V's own has `K_enc`, genuinely
    # different row counts. Mirrors test_attention_head_pruning_cpp.py's own
    # `_gqa_cross_model` (and test_pruning.py's own).
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    body = f"""
        g (float[{batch},{seq},{K_dec}] Xdec, float[{batch},{seq},{K_enc}] Xenc) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_wanda_pruning_cross_attention_matches_oracle_exactly():
    # Regression coverage for `ApplyOneGqaChain`'s own `Kq`/`Kk`/`Kv` fix:
    # without it, a single shared row count (Q's own) reused to index into
    # `wk_kn`/`wv_kn` too is both wrong and an out-of-bounds read whenever
    # K/V's producer has a different row count than Q's -- K_dec=8 !=
    # K_enc=6 here is deliberate. The Wanda-calibrated activation-norm
    # multiply shares this exact same importance computation (see
    # `ApplyOneGqaChain`'s own doc comment), so this closes the identical bug
    # for the calibrated path too.
    model, cfg = _gqa_cross_model(K_dec=8, K_enc=6, H=8, KVH=2, D=8, Out=6, seed=22)

    rng = np.random.default_rng(23)
    xdec_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    calibration_data = [{"Xdec": xdec_cal, "Xenc": xenc_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_gqa_wanda_pruning_cross_attention_matches_oracle_exactly"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    act_norm = _probe_act_norm(model, "ctx", {"Xdec": xdec_cal, "Xenc": xenc_cal})

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.sqrt(
            np.linalg.norm(q_block) ** 2
            + np.linalg.norm(k_block) ** 2
            + np.linalg.norm(v_block) ** 2
        )
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=22,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    xdec = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


# --- Packed-QKV-then-Split + RoPE/QK-norm walk-back, Wanda-calibrated -------
# --- (GroupQueryAttention/plain ai.onnx::Attention) -- mirrors
# --- test_attention_head_pruning_cpp.py's own identically-named section,
# --- confirming FindSeparateQkvChains's new WalkBackThroughQkNormRope/
# --- WalkBackThroughGemmaRopePair machinery is recognized under the
# --- Wanda-calibrated path too, not just the data-free default (both share
# --- FindGqaChains/FindOnnxAttentionChains verbatim -- only
# --- compute_group_importance differs). Model builders duplicated here
# --- (rather than imported), matching this test file's own established
# --- per-file-duplicate convention (e.g. `_gqa_model` above).


def _gqa_packed_model_w(K=8, H=4, KVH=2, D=8, Out=6, seed=0, batch=2, seq=5):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [
        onnx.numpy_helper.from_array(wqkv, "Wqkv"),
        onnx.numpy_helper.from_array(wout, "Wout"),
    ]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q, k, v = Split <axis = -1> (qkv, SplitSizes)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(K=K, H=H, KVH=KVH, D=D, Out=Out, batch=batch, seq=seq)


def test_cpp_gqa_wanda_packed_qkv_split_pruning_matches_python_reference():
    model, cfg = _gqa_packed_model_w(K=8, H=8, KVH=2, D=8, Out=6, seed=401)
    rng_cal = np.random.default_rng(402)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_gqa_wanda_packed_qkv_split_pruning_matches_python_reference"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def _qk_norm_rope_body_w(
    prefix,
    raw_name,
    gamma_name,
    reshape1_shape_name,
    reshape2_shape_name,
    with_norm,
    with_rope,
    cos_name="CosCache",
    sin_name="SinCache",
    pos_name="PosIds",
    rotary_num_heads=None,
):
    lines = []
    cur = raw_name
    if with_norm:
        lines.append(f"{prefix}_r1 = Reshape({cur}, {reshape1_shape_name})")
        lines.append(
            f"{prefix}_ln = SimplifiedLayerNormalization <axis=-1, epsilon=1e-6> "
            f"({prefix}_r1, {gamma_name})"
        )
        lines.append(f"{prefix}_normed = Reshape({prefix}_ln, {reshape2_shape_name})")
        cur = f"{prefix}_normed"
    if with_rope:
        nh = rotary_num_heads if rotary_num_heads is not None else 0
        lines.append(
            f"{prefix}_rot = com.microsoft.RotaryEmbedding <num_heads={nh}> "
            f"({cur}, {pos_name}, {cos_name}, {sin_name})"
        )
        cur = f"{prefix}_rot"
    return "\n          ".join(lines), cur


def _gqa_qk_norm_rope_model_w(
    K=8, H=8, KVH=2, D=8, Out=6, seed=0, batch=2, seq=5, max_pos=32
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [
        onnx.numpy_helper.from_array(wqkv, "Wqkv"),
        onnx.numpy_helper.from_array(wout, "Wout"),
    ]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    half = D // 2
    cos = rng.standard_normal((max_pos, half)).astype(np.float32)
    sin = rng.standard_normal((max_pos, half)).astype(np.float32)
    position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    initializer += [
        onnx.numpy_helper.from_array(cos, "CosCache"),
        onnx.numpy_helper.from_array(sin, "SinCache"),
    ]
    initializer.append(onnx.numpy_helper.from_array(position_ids, "PosIds"))
    q_gamma = rng.standard_normal((D,)).astype(np.float32)
    k_gamma = rng.standard_normal((D,)).astype(np.float32)
    initializer += [
        onnx.numpy_helper.from_array(q_gamma, "QGamma"),
        onnx.numpy_helper.from_array(k_gamma, "KGamma"),
    ]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, D], dtype=np.int64), "QReshape1Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, Nq], dtype=np.int64), "QReshape2Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, D], dtype=np.int64), "KReshape1Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, Nkv], dtype=np.int64), "KReshape2Shape"
        )
    )
    q_text, q_body = _qk_norm_rope_body_w(
        "q", "q_raw", "QGamma", "QReshape1Shape", "QReshape2Shape", True, True
    )
    k_text, k_body = _qk_norm_rope_body_w(
        "k", "k_raw", "KGamma", "KReshape1Shape", "KReshape2Shape", True, True
    )
    hop_body = q_text + "\n          " + k_text
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          {hop_body}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({q_body}, {k_body}, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(K=K, H=H, KVH=KVH, D=D, Out=Out, batch=batch, seq=seq)


def test_cpp_gqa_wanda_packed_qkv_qk_norm_rope_pruning_matches_python_reference():
    model, cfg = _gqa_qk_norm_rope_model_w(K=8, H=8, KVH=2, D=8, Out=6, seed=403)
    rng_cal = np.random.default_rng(404)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_gqa_wanda_packed_qkv_qk_norm_rope_pruning_matches_python_reference"
        ]
    )
    # No onnx.checker.check_model here -- `SimplifiedLayerNormalization`
    # registers under the default ("") domain but isn't in onnx's own schema
    # registry at all, so the checker declines this exact op/domain
    # combination regardless of this port's own correctness -- see
    # test_attention_head_pruning_cpp.py's own identical note.
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def _onnx_attention_packed_rope_model_w(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    q_rotary_num_heads=None,
    k_rotary_num_heads=None,
    max_pos=32,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [
        onnx.numpy_helper.from_array(wqkv, "Wqkv"),
        onnx.numpy_helper.from_array(wout, "Wout"),
    ]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    half = D // 2
    cos = rng.standard_normal((max_pos, half)).astype(np.float32)
    sin = rng.standard_normal((max_pos, half)).astype(np.float32)
    position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    initializer += [
        onnx.numpy_helper.from_array(cos, "CosCache"),
        onnx.numpy_helper.from_array(sin, "SinCache"),
    ]
    initializer.append(onnx.numpy_helper.from_array(position_ids, "PosIds"))
    qnh = q_rotary_num_heads if q_rotary_num_heads is not None else 0
    knh = k_rotary_num_heads if k_rotary_num_heads is not None else 0
    hop_body = (
        f"q = RotaryEmbedding <num_heads={qnh}> (q_raw, CosCache, SinCache, PosIds)\n"
        f"          k = RotaryEmbedding <num_heads={knh}> (k_raw, CosCache, SinCache, PosIds)"
    )
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          {hop_body}
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(K=K, H=H, KVH=KVH, D=D, Out=Out, batch=batch, seq=seq)


def test_cpp_onnx_attention_wanda_packed_qkv_native_rotary_embedding_pruning_matches_python_reference():
    model, cfg = _onnx_attention_packed_rope_model_w(
        K=8,
        H=8,
        KVH=2,
        D=8,
        Out=6,
        seed=405,
        q_rotary_num_heads=8,
        k_rotary_num_heads=2,
    )
    rng_cal = np.random.default_rng(406)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_onnx_attention_wanda_packed_qkv_native_rotary_embedding_pruning_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- True MQA (kv_num_heads == 1) fused GroupQueryAttention Wanda fast path -
#
# Wanda-calibrated counterpart of test_attention_head_pruning_cpp.py's own
# true-MQA `GroupQueryAttention` fast-path coverage: `ApplyOneGqaChain`'s own
# `act_norm` branch is shared, unmodified, between the plain and Wanda entry
# points (`apply_attention_head_pruning_cpp` passes `nullptr`,
# `apply_attention_head_wanda_pruning_cpp` a real calibrated map), so the
# same `is_mqa` fix that closes the plain no-op bug for `kv_num_heads == 1`
# closes it here too, ranking each query head by its own weight norm scaled
# by its own `d`-wide slice of the calibrated activation (mirroring
# pruning.py's own `_wanda_gqa_query_head_importance`).


def _oracle_wanda_keep_query_heads(wq, num_heads, head_size, act_norm, keep_count):
    importance = np.zeros(num_heads)
    for h in range(num_heads):
        block_norm = np.linalg.norm(wq[:, h * head_size : (h + 1) * head_size])
        act_head = np.linalg.norm(act_norm[h * head_size : (h + 1) * head_size])
        importance[h] = block_norm * max(act_head, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_mqa_wanda_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=1, D=8, Out=6, seed=15)

    rng = np.random.default_rng(16)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep_q_heads = _oracle_wanda_keep_query_heads(
        cfg["wq"], cfg["H"], cfg["D"], act_norm, 4
    )

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == len(keep_q_heads)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])

    d = cfg["D"]
    q_idx = _head_idx(keep_q_heads, d)
    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=1,
        D=d,
        Out=cfg["Out"],
        seed=15,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"],
        wv=cfg["wv"],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_attention_head_wanda_pruning_matches_python_reference_plain():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=80)
    rng_cal = np.random.default_rng(81)
    x_cal = rng_cal.standard_normal((3, 6, 8)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_attention_head_wanda_pruning_matches_python_reference_plain"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_attention_head_wanda_pruning_matches_python_reference_multi_batch():
    # Multiple calibration batches, accumulated sum-of-squares across all of
    # them -- exercises WandaCalibrationStats' own per-batch accumulation
    # loop against pruning.py's own identical `for batch in calibration_data`
    # loop.
    model, _ = _attention_model(K=6, H=4, D=2, Out=5, seed=90, batch=2, seq=4)
    rng_cal = np.random.default_rng(91)
    calibration_data = [
        {"X": rng_cal.standard_normal((2, 4, 6)).astype(np.float32)} for _ in range(4)
    ]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_attention_head_wanda_pruning_matches_python_reference_multi_batch"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_gqa_wanda_pruning_matches_python_reference():
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=100)
    rng_cal = np.random.default_rng(101)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(_GOLDEN["test_cpp_gqa_wanda_pruning_matches_python_reference"])
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- Error handling ----------------------------------------------------------


def test_cpp_attention_head_wanda_pruning_missing_calibration_input_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=120)
    bad_batch = {"NotX": np.zeros((2, 5, 8), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[bad_batch], sparsity=0.5
        )


def test_cpp_attention_head_wanda_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=121)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=1.0
        )
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=-0.1
        )


# --- Default (auto-generated) calibration data ------------------------------


def test_cpp_attention_head_wanda_pruning_default_calibration_data_runs():
    # calibration_data=None generates random calibration batches via
    # onnxsim.generate_random_calibration_data (symbolic batch/seq dims
    # fixed to 1), matching the pure-Python
    # apply_attention_head_wanda_pruning's own default -- just confirms the
    # whole path runs end to end and produces a valid, actually-pruned
    # model, not a specific oracle (random data has no fixed oracle here).
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=130)
    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, num_samples=4, seed=5, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2


# --- importance_norm ("l1" vs "l2") ------------------------------------------
#
# Same adversarial per-head/per-group weight blocks as
# test_attention_head_pruning_cpp.py's own importance_norm tests, driven
# through the *empty-calibration-data* fallback path (mirrors
# `test_cpp_attention_head_wanda_pruning_empty_calibration_data_matches_plain`/
# `test_cpp_gqa_wanda_pruning_empty_calibration_data_matches_plain` above):
# with no observed activation, `_wanda_attention_head_importance`/
# `_wanda_gqa_group_importance` fall straight back to the plain
# `||W||`-only ranking, so this isolates the *weight*-magnitude term's own
# L1-vs-L2 switch from the (always-L2) activation-norm term -- while still
# exercising the real Wanda entry point/binding end to end, not just the
# plain one.


def test_cpp_attention_head_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, D, Out = 16, 4, 4, 3
    Nq = Nk = Nv = H * D
    rng_qk = np.random.default_rng(52)
    wqkv = np.zeros((K, Nq + Nk + Nv), dtype=np.float32)
    wqkv[:, :Nq] = rng_qk.standard_normal((K, Nq)).astype(np.float32) * 0.01
    wqkv[:, Nq : Nq + Nk] = rng_qk.standard_normal((K, Nk)).astype(np.float32) * 0.01
    v_offset = Nq + Nk
    wqkv[0, v_offset + 0] = 16.0  # head 0 ("concentrated")
    wqkv[:, v_offset + D : v_offset + 2 * D] = 1.0  # head 1 ("spread")
    wqkv[2, v_offset + 2 * D] = 1000.0  # head 2 ("filler_high")
    wqkv[3, v_offset + 3 * D] = 0.001  # head 3 ("filler_low")
    bqkv = np.zeros((Nq + Nk + Nv,), dtype=np.float32)

    model, _cfg = _attention_model(
        K=K, H=H, D=D, Out=Out, seed=50, bias=True, wqkv=wqkv, bqkv=bqkv
    )

    golden = _GOLDEN[
        "test_cpp_attention_head_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2"
    ]
    for i, norm in enumerate(("l2", "l1")):
        pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        pruned_py = _golden(golden[i])
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    kept_l1 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5, importance_norm="l1"
    )
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def test_cpp_gqa_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, KVH, D, Out = 8, 4, 2, 8, 3
    Nq, Nkv = H * D, KVH * D
    wq = np.zeros((K, Nq), dtype=np.float32)
    wk = np.zeros((K, Nkv), dtype=np.float32)
    wv = np.zeros((K, Nkv), dtype=np.float32)
    wv[0, 0] = 16.0  # KV group 0's own V slice -- concentrated
    wv[:, D : 2 * D] = 1.0  # KV group 1's own V slice -- spread

    model, _cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=60, wq=wq, wk=wk, wv=wv
    )

    golden = _GOLDEN[
        "test_cpp_gqa_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2"
    ]
    for i, norm in enumerate(("l2", "l1")):
        pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        pruned_py = _golden(golden[i])
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    kept_l1 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5, importance_norm="l1"
    )
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


# --- Plain ai.onnx::Attention (opset 24+) -----------------------------------


def _onnx_attention_model(
    K=8,
    H=8,
    KVH=2,
    D=4,
    Dv=None,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    # `Dv` (V's own head_size, defaulting to `D`) is independent of Q/K's
    # `D` -- this op's own schema genuinely allows the two to differ (see
    # test_attention_head_pruning_cpp.py's own `_onnx_attention_model`, which
    # this mirrors).
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Dv=Dv,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _onnx_attention_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "Attention" and n.domain == ""
    )


def _onnx_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_cpp_onnx_attention_wanda_pruning_diff_v_head_size_matches_oracle_exactly():
    # The Wanda-calibrated counterpart of
    # test_attention_head_pruning_cpp.py's own
    # `test_cpp_onnx_attention_pruning_diff_v_head_size_matches_oracle_exactly`:
    # the activation probe sits on the consumer's input (the attention
    # output), laid out per query head at V's own `v_head_size` -- not Q's/
    # K's `head_size` -- so both the width check and the per-group activation
    # window must stride by `v_head_size`, exactly like the weight-only
    # path's `y_idx` does.
    K, H, KVH, D, Dv, Out = 8, 8, 2, 4, 6, 5
    group_size = H // KVH
    model, cfg = _onnx_attention_model(K=K, H=H, KVH=KVH, D=D, Dv=Dv, Out=Out, seed=23)

    rng = np.random.default_rng(24)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_onnx_attention_wanda_pruning_diff_v_head_size_matches_oracle_exactly"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    assert act_norm.shape == (H * Dv,)  # sanity: laid out per Q head at Dv, not D

    importance = np.zeros(KVH)
    for kv in range(KVH):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * D : (h + 1) * D]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * D : (kv + 1) * D]
        v_block = cfg["wv"][:, kv * Dv : (kv + 1) * Dv]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * Dv : (kv + 1) * group_size * Dv]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5))

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx = _head_idx(keep_q_heads, D)
    kv_idx = _head_idx(keep_groups, D)
    v_idx = _head_idx(keep_groups, Dv)
    y_idx = _head_idx(keep_q_heads, Dv)

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, v_idx])
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][y_idx, :])


# ============================================================================
# Six more matched families -- see test_attention_head_pruning_cpp.py's own
# identical section for the model builders' full rationale and this port's
# deliberate, narrower-than-pruning.py scope decisions (no dynamic-
# attention-bias-Gather-insertion machinery, so every new matcher declines
# outright whenever such an optional input resolves to a non-empty
# constant). Every "matches python reference" test below cross-checks
# against ``onnxsim.apply_attention_head_wanda_pruning`` (the pure-Python
# reference) run with the SAME real calibration data through the SAME
# ``onnxruntime``-backed executor, so a difference here would mean the two
# ports' calibration-crossing/importance-ranking logic genuinely disagree,
# not merely "both produce a valid model".


def _mha_model(
    K=8, H=4, D=4, Out=6, seed=0, batch=2, seq=5, wq=None, wk=None, wv=None, wout=None
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.MultiHeadAttention <num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "MultiHeadAttention")


def _mha_num_heads(node):
    return next(a.i for a in node.attribute if a.name == "num_heads")


def _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size):
    group_size = num_heads // kv_num_heads
    importance = np.zeros(kv_num_heads)
    for kv in range(kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * head_size : (h + 1) * head_size]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * head_size : (kv + 1) * head_size]
        v_block = wv[:, kv * head_size : (kv + 1) * head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return importance


def _wanda_gqa_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, act_norm, keep_count
):
    group_size = num_heads // kv_num_heads
    base = _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size)
    act_group = np.array(
        [
            np.linalg.norm(
                act_norm[
                    kv * group_size * head_size : (kv + 1) * group_size * head_size
                ]
            )
            for kv in range(kv_num_heads)
        ]
    )
    importance = base * np.maximum(act_group, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_mha_wanda_pruning_matches_oracle_exactly():
    model, cfg = _mha_model(K=8, H=8, D=4, Out=6, seed=200)
    rng = np.random.default_rng(201)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep_heads = _wanda_gqa_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["H"], cfg["D"], act_norm, 4
    )

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    node = _mha_node(pruned)
    assert _mha_num_heads(node) == len(keep_heads)

    d = cfg["D"]
    idx = _head_idx(keep_heads, d)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, idx])
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][idx, :])


def test_cpp_mha_wanda_pruning_matches_python_reference():
    model, cfg = _mha_model(K=8, H=8, D=4, Out=6, seed=202)
    rng = np.random.default_rng(203)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(_GOLDEN["test_cpp_mha_wanda_pruning_matches_python_reference"])
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- com.microsoft::PackedMultiHeadAttention --------------------------------


def _packed_mha_model(
    K=8, H=4, D=4, Out=6, seed=0, tok=5, wq=None, wk=None, wv=None, wout=None
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    token_offset = np.arange(tok, dtype=np.int32).reshape(1, tok)
    cum_seq_len = np.array([0, tok], dtype=np.int32)
    initializer = [
        _f32(wq, "Wq"),
        _f32(wk, "Wk"),
        _f32(wv, "Wv"),
        _f32(wout, "Wout"),
        onnx.numpy_helper.from_array(token_offset, "TokenOffset"),
        onnx.numpy_helper.from_array(cum_seq_len, "CumSeqLen"),
    ]
    body = f"""
        g (float[{tok},{K}] X) => (float[{tok},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.PackedMultiHeadAttention <num_heads={H}> (q, k, v, , TokenOffset, CumSeqLen)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        tok=tok,
    )


def _packed_mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "PackedMultiHeadAttention")


def test_cpp_packed_mha_wanda_pruning_empty_calibration_data_matches_python_reference():
    # No CPU kernel exists for `PackedMultiHeadAttention` in this environment
    # (confirmed via a real `onnxruntime.InferenceSession` load -- see
    # `_match_packed_multi_head_attention_producer`'s own docstring and this
    # file's sibling `test_attention_head_pruning_cpp.py`'s own note on the
    # same op) -- so a *real* calibration run (which must actually execute
    # the graph) can never succeed for this op on any input, Python or C++
    # port alike. `calibration_data=[]` exercises the real Wanda entry point
    # end to end without ever executing the graph (falls back to plain
    # `||W||_F` ranking, the same "no observed activation" fallback every
    # other family's own analogous empty-calibration-data test already
    # relies on), while still cross-checking this port's own combined-bias/
    # importance-ranking logic against the Python reference exactly.
    model, _ = _packed_mha_model(K=8, H=8, D=4, Out=6, seed=210)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_packed_mha_wanda_pruning_empty_calibration_data_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


# --- com.microsoft::DecoderMaskedMultiHeadAttention -------------------------


def _dmmha_model(
    K=8, H=4, D=4, Out=6, seed=0, batch=2, wq=None, wk=None, wv=None, wout=None
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    body = f"""
        g (float[{batch},1,{K}] X) => (float[{batch},1,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.DecoderMaskedMultiHeadAttention <num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
    )


def _dmmha_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "DecoderMaskedMultiHeadAttention"
    )


def test_cpp_dmmha_wanda_pruning_matches_python_reference():
    model, cfg = _dmmha_model(K=8, H=8, D=4, Out=6, seed=220)
    rng = np.random.default_rng(221)
    x_cal = rng.standard_normal((cfg["batch"], 1, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_dmmha_wanda_pruning_matches_python_reference"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- com.microsoft::PagedAttention -------------------------------------------
#
# See test_attention_head_pruning_cpp.py's own identical section for why
# this deliberately builds a float32 (not the real schema's float16-only)
# model -- the shared Q/K/V-producer-matching machinery this whole "Attention
# -head pruning" C++ section uses is FLOAT32-only, a pre-existing restriction
# shared by every family, GQA/OnnxAttention included.


def _paged_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    num_tokens=3,
    num_blocks=2,
    block_size=4,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.5
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.5
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.5
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32) * 0.5
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, num_tokens], dtype=np.int32), "CumSeqLen"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.zeros((1,), dtype=np.int32), "PastSeqLens")
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.zeros((1, 1), dtype=np.int32), "BlockTable")
    )
    extra_inputs = (
        f", float[{num_blocks},{block_size},{KVH},{D}] KeyCache"
        f", float[{num_blocks},{block_size},{KVH},{D}] ValueCache"
    )
    operands = [
        "q",
        "k",
        "v",
        "KeyCache",
        "ValueCache",
        "CumSeqLen",
        "PastSeqLens",
        "BlockTable",
    ]
    body = f"""
        g (float[{num_tokens},{K}] X{extra_inputs}) => (float[{num_tokens},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, key_cache_out, value_cache_out = com.microsoft.PagedAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        num_tokens=num_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
    )


def _paged_node(model):
    return next(n for n in model.graph.node if n.op_type == "PagedAttention")


def test_cpp_paged_attention_wanda_pruning_empty_calibration_data_matches_python_reference():
    # This op's real onnxruntime schema requires `query`/`key`/`value` be
    # float16/bfloat16 (see this file's sibling
    # `test_attention_head_pruning_cpp.py`'s own note on this op) -- so a
    # float32 model, the only kind this port's shared FLOAT32-only Q/K/V-
    # producer-matching machinery can match at all, fails outright at
    # `InferenceSession` graph-load time ("Type Error: Type 'tensor(float)'
    # ... is invalid"), confirmed empirically. A *real* calibration run
    # (which must actually execute the graph) can therefore never succeed
    # for this port's own matched shape -- `calibration_data=[]` exercises
    # the real Wanda entry point end to end without ever executing the
    # graph (falls back to plain `||W||_F` ranking), still cross-checking
    # this port's own importance-ranking/slicing logic against the Python
    # reference exactly.
    model, _ = _paged_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=230)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_paged_attention_wanda_pruning_empty_calibration_data_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


# --- Plain ai.onnx::LinearAttention (opset 27+, "linear" update_rule only) --


def _linear_attention_model(
    Hq=4, Hkv=2, D=4, K=16, seed=0, wq=None, wk=None, wv=None, wo=None
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = Hq * D, Hkv * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.3
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.3
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.3
    if wo is None:
        wo = rng.standard_normal((Nq, K)).astype(np.float32) * 0.3
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wo, "Wo")]
    body = f"""
        g (float[1,3,{K}] X) => (float[1,3,{K}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          attn_out, ps = LinearAttention<q_num_heads={Hq}, kv_num_heads={Hkv}, update_rule="linear">(q, k, v)
          Y = MatMul(attn_out, Wo)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 27]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(Hq=Hq, Hkv=Hkv, D=D, K=K, wq=wq, wk=wk, wv=wv, wo=wo)


def _linear_attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "LinearAttention")


def test_cpp_linear_attention_wanda_pruning_matches_python_reference():
    # `LinearAttention` is plain ai.onnx opset 27, which this environment's
    # onnxruntime treats as "under development" and refuses to load by
    # default (`ValidateOpsetForDomain`) -- a real calibration run must
    # actually execute the graph, so this relaxes that load-time check the
    # same way `test_pruning.py`'s own `_run27` helper does (see that
    # helper's own comment); it does not stub out or change this op's own
    # real CPU kernel.
    os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
    model, cfg = _linear_attention_model(Hq=8, Hkv=4, D=4, K=16, seed=240)
    rng = np.random.default_rng(241)
    x_cal = rng.standard_normal((1, 3, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN["test_cpp_linear_attention_wanda_pruning_matches_python_reference"]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- com.microsoft::SparseAttention ------------------------------------------


def _sparse_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=1,
    seq=16,
    sparse_block_size=16,
    num_layout=1,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    past_kv="dynamic",  # "dynamic" (graph input only) | "nonempty" (constant, sliced)
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.1
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.1
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.1
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32) * 0.1
    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    row_indices = np.tile(np.array([0, 1], dtype=np.int32), (num_layout, 1))
    col_indices = np.tile(np.array([0], dtype=np.int32), (num_layout, 1))
    initializer.append(onnx.numpy_helper.from_array(row_indices, "RowIdx"))
    initializer.append(onnx.numpy_helper.from_array(col_indices, "ColIdx"))
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq, dtype=np.int32), "KeyTotalSeq"
        )
    )
    extra_inputs = ""
    past_key = past_value = None
    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, seq, D)).astype(np.float32) * 0.1
        past_value = rng.standard_normal((batch, KVH, seq, D)).astype(np.float32) * 0.1
        initializer += [
            onnx.numpy_helper.from_array(past_key, "PastKey"),
            onnx.numpy_helper.from_array(past_value, "PastValue"),
        ]
    else:
        extra_inputs = (
            f", float[{batch},{KVH},{seq},{D}] PastKey"
            f", float[{batch},{KVH},{seq},{D}] PastValue"
        )
    body = f"""
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          attn, PresentKey, PresentValue = com.microsoft.SparseAttention <num_heads={H}, kv_num_heads={KVH}, sparse_block_size={sparse_block_size}> (q, k, v, PastKey, PastValue, RowIdx, ColIdx, TotalSeq, KeyTotalSeq)
          Y = MatMul(attn, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
        past_key=past_key,
        past_value=past_value,
    )


def _sparse_attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "SparseAttention")


def _sparse_attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def test_cpp_sparse_attention_wanda_pruning_empty_calibration_data_matches_python_reference():
    # The real onnxruntime CPU kernel for `SparseAttention` requires
    # `past_key`/`past_value` be bound to the EXACT SAME buffer as
    # `present_key`/`present_value` (`past_key->DataRaw() ==
    # present_key->DataRaw()`, confirmed empirically -- see this file's
    # sibling `test_pruning.py`'s own `_run_sparse_attention` helper, which
    # exists specifically to satisfy this via `onnxruntime`'s IOBinding
    # API). Plain `sess.run()` -- what this port's (and pruning.py's own)
    # Wanda calibration machinery uses to probe activations -- can never
    # satisfy that in-place-update requirement, so a *real* calibration run
    # can never succeed for this op regardless of pruning correctness.
    # `calibration_data=[]` exercises the real Wanda entry point end to end
    # without ever executing the graph (falls back to plain `||W||_F`
    # ranking), still cross-checking this port's own num_layout-divisibility/
    # importance-ranking/slicing logic against the Python reference exactly.
    model, _ = _sparse_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=250)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_sparse_attention_wanda_pruning_empty_calibration_data_matches_python_reference"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


def test_cpp_sparse_attention_wanda_pruning_nonempty_past_kv_constant_is_sliced_matches_python():
    # `SparseAttention`'s own required `past_key`/`past_value` now go through
    # the exact same `PastKvConstantsAreSliceable`/`SliceKvCacheAxis1`
    # validate-and-slice machinery whether reached from the plain
    # (`apply_attention_head_pruning_cpp`, see
    # ``test_attention_head_pruning_cpp.py``'s own dedicated coverage) or this
    # Wanda-calibrated entry point -- both dispatch through the identical
    # `MatchSparseAttentionProducer`/`ApplyOneGqaChain`. `calibration_data=[]`
    # (see this file's sibling test above for why a REAL calibration run can
    # never succeed for this op) still exercises this fix end to end through
    # this entry point's own separate graph/used_names/value_info_by_name
    # plumbing.
    model, cfg = _sparse_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=251, past_kv="nonempty"
    )
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_sparse_attention_wanda_pruning_nonempty_past_kv_constant_is_sliced_matches_python"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _sparse_attention_node(pruned_cpp)
    _, kv_num_heads = _sparse_attention_attrs(node)
    assert kv_num_heads == 1
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert list(inits["PastKey"].shape) == [
        cfg["batch"],
        kv_num_heads,
        cfg["seq"],
        cfg["D"],
    ]


# --- Decomposed (un-fused) GQA/MQA/plain-MHA attention head pruning --------
#
# `apply_attention_head_wanda_pruning_cpp` now also matches this shape, via
# the same genuinely new, dedicated FindDecomposedGqaChains/
# ApplyOneDecomposedGqaChain machinery `apply_attention_head_pruning_cpp`
# uses (threaded through with a real calibrated `act_norm` map, exactly like
# every other family here) -- see
# ``test_attention_head_pruning_cpp.py``'s own "Decomposed (un-fused)
# GQA/MQA/plain-MHA attention head pruning" section comment for the exact
# scope this port matches (deliberately narrower than pruning.py's own
# ``_find_decomposed_gqa_chains``: no mask/RoPE/Q-K-norm/Einsum) -- packed-
# QKV-then-``Split`` producers and true-MQA ARE matched/applied here, via
# the same shared machinery -- so this tenth family is NOT yet aliased
# either.


def _decomposed_gqa_model(
    K=32,
    H=4,
    KVH=2,
    D=8,
    Dv=None,
    Out=16,
    batch=1,
    seq=4,
    seed=0,
    bias=True,
    masked=False,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    bq=None,
    bk=None,
    bv=None,
    bout=None,
    share_kv_reshape_shape=True,
    extra_foreign_q_reshape_consumer=False,
):
    """Builds the decomposed-attention graph FindDecomposedGqaChains matches
    -- mirrors ``test_attention_head_pruning_cpp.py``'s own
    ``_decomposed_gqa_model`` (trimmed further: no ``out_reshape_wildcard``/
    ``extra_foreign_repeat_kv_consumer`` params, not needed by this file's
    own coverage -- see that copy's own docstring for the full shape and
    every other parameter's meaning)."""
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]

    initializer.append(_i64([batch * seq, K], "XFlatShape"))
    lines = ["xf = Reshape(X, XFlatShape)"]
    q_op, k_op, v_op, o_op = (
        "MatMul(xf, Wq)",
        "MatMul(xf, Wk)",
        "MatMul(xf, Wv)",
        "MatMul(ctx2, Wout)",
    )
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        if bout is None:
            bout = rng.standard_normal((Out,)).astype(np.float32)
        initializer += [
            _f32(bq, "Bq"),
            _f32(bk, "Bk"),
            _f32(bv, "Bv"),
            _f32(bout, "Bout"),
        ]
        q_op, k_op, v_op = "Gemm(xf, Wq, Bq)", "Gemm(xf, Wk, Bk)", "Gemm(xf, Wv, Bv)"
        o_op = "Gemm(ctx2, Wout, Bout)"

    initializer.append(_i64([batch, seq, H, D], "Sq"))
    lines += [
        "q0 = " + q_op,
        "qr = Reshape(q0, Sq)",
        "qt = Transpose<perm=[0,2,1,3]>(qr)",
    ]

    if extra_foreign_q_reshape_consumer:
        lines.append("foreign_out = Reshape(xf, Sq)")

    kv_shape_name = "Skv" if share_kv_reshape_shape and Dv == D else None
    if kv_shape_name:
        initializer.append(_i64([batch, seq, KVH, D], kv_shape_name))
        sk_name = sv_name = kv_shape_name
    else:
        initializer.append(_i64([batch, seq, KVH, D], "Sk"))
        initializer.append(_i64([batch, seq, KVH, Dv], "Sv"))
        sk_name, sv_name = "Sk", "Sv"

    lines += ["k0 = " + k_op, f"kr = Reshape(k0, {sk_name})"]
    lines += ["v0 = " + v_op, f"vr = Reshape(v0, {sv_name})"]

    n_rep = H // KVH
    needs_repeat_kv = KVH < H
    if needs_repeat_kv:
        assert H % KVH == 0
        initializer.append(_i64([2], "Ax2"))
        initializer.append(_i64([batch, KVH, n_rep, seq, D], "KExpandShape"))
        initializer.append(_i64([batch, H, seq, D], "KMergeShape"))
        initializer.append(_i64([batch, KVH, n_rep, seq, Dv], "VExpandShape"))
        initializer.append(_i64([batch, H, seq, Dv], "VMergeShape"))
        lines.append("kt0 = Transpose<perm=[0,2,1,3]>(kr)")
        lines += [
            "ku = Unsqueeze(kt0, Ax2)",
            "ke = Expand(ku, KExpandShape)",
            "kre = Reshape(ke, KMergeShape)",
            "kt = Transpose<perm=[0,1,3,2]>(kre)",
            "vt0 = Transpose<perm=[0,2,1,3]>(vr)",
            "vu = Unsqueeze(vt0, Ax2)",
            "ve = Expand(vu, VExpandShape)",
            "vt = Reshape(ve, VMergeShape)",
        ]
    else:
        lines.append("kt = Transpose<perm=[0,2,3,1]>(kr)")
        lines.append("vt = Transpose<perm=[0,2,1,3]>(vr)")

    initializer.append(_f32(np.array(D**-0.5, dtype=np.float32), "Scale"))
    lines += ["qk = MatMul(qt, kt)", "scaled = Mul(qk, Scale)"]

    if masked:
        mask = np.triu(np.full((seq, seq), -1e4, dtype=np.float32), k=1)[
            None, None, :, :
        ]
        initializer.append(_f32(mask, "Mask"))
        lines.append("premask = Add(scaled, Mask)")
        smax_in = "premask"
    else:
        smax_in = "scaled"
    lines.append(f"attn = Softmax<axis=-1>({smax_in})")
    lines.append("ctx0 = MatMul(attn, vt)")
    lines.append("ctx1 = Transpose<perm=[0,2,1,3]>(ctx0)")

    initializer.append(_i64([batch * seq, H * Dv], "OutShape"))
    initializer.append(_i64([batch, seq, Out], "YShape"))
    lines.append("ctx2 = Reshape(ctx1, OutShape)")
    lines.append("y0 = " + o_op)
    lines.append("Y = Reshape(y0, YShape)")

    body_lines = "\n          ".join(lines)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          {body_lines}
        }}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Dv=Dv,
        Out=Out,
        batch=batch,
        seq=seq,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        bq=bq,
        bk=bk,
        bv=bv,
        bout=bout,
    )


def _decomposed_weight_shapes(model):
    inits = {t.name: t for t in model.graph.initializer}
    return (
        onnx.numpy_helper.to_array(inits["Wq"]),
        onnx.numpy_helper.to_array(inits["Wk"]),
        onnx.numpy_helper.to_array(inits["Wv"]),
        onnx.numpy_helper.to_array(inits["Wout"]),
    )


def test_cpp_decomposed_gqa_wanda_pruning_matches_oracle_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=101)
    rng = np.random.default_rng(102)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx2", {"X": x_cal})
    new_kv = 1  # KVH=2, sparsity=0.5 -> keep_count = max(1, 2 - round(1)) == 1
    keep_groups = _wanda_gqa_keep_groups(
        cfg["wq"],
        cfg["wk"],
        cfg["wv"],
        cfg["H"],
        cfg["KVH"],
        cfg["D"],
        act_norm,
        new_kv,
    )
    group_size = cfg["H"] // cfg["KVH"]
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])


def test_cpp_decomposed_gqa_wanda_pruning_matches_python_reference_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=103)

    rng = np.random.default_rng(104)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
        for _ in range(3)
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_decomposed_gqa_wanda_pruning_matches_python_reference_exactly"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_decomposed_gqa_wanda_pruning_clones_shape_constant_shared_with_foreign_reader():
    model, cfg = _decomposed_gqa_model(
        K=64,  # == H * D, so `foreign_out`'s own Reshape(xf, Sq) is valid.
        H=8,
        KVH=2,
        D=8,
        Out=16,
        seed=105,
        extra_foreign_q_reshape_consumer=True,
    )

    rng = np.random.default_rng(106)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_decomposed_gqa_wanda_pruning_clones_shape_constant_shared_with_foreign_reader"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(
        inits["Sq"], [cfg["batch"], cfg["seq"], cfg["H"], cfg["D"]]
    )
    assert "Sq_pruned" in inits


def test_cpp_decomposed_gqa_wanda_pruning_with_constant_mask_matches_python_reference_exactly():
    # An additive mask `Add` before `Softmax`, a constant of the
    # schema-documented per-head-broadcastable shape -- `ApplyAttentionHead
    # WandaPruning` shares `FindDecomposedGqaChains`/`ApplyOneDecomposedGqaChain`
    # with the data-free entry point above, so it now matches and correctly
    # leaves this broadcast mask untouched here too (see
    # test_attention_head_pruning_cpp.py's own identical, more thorough
    # coverage of this branch for the full reasoning) -- both ports must
    # agree byte-for-byte, whether or not calibration data is supplied.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=107, masked=True
    )
    rng = np.random.default_rng(108)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_decomposed_gqa_wanda_pruning_with_constant_mask_matches_python_reference_exactly"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def _wanda_query_head_keep(wq, d, num_heads, act_norm, dv, keep_count, epsilon=1e-8):
    # Mirrors pruning.py's own `_wanda_gqa_query_head_importance`: each
    # query head's own weight-only score (`_gqa_query_head_importance`),
    # scaled by that SAME head's own `dv`-wide slice of the probed
    # activation (falling back to the plain weight-only score when the
    # probed width doesn't match).
    base = np.array(
        [np.linalg.norm(wq[:, h * d : (h + 1) * d]) for h in range(num_heads)]
    )
    if act_norm.shape[0] == num_heads * dv:
        act_head = np.array(
            [np.linalg.norm(act_norm[h * dv : (h + 1) * dv]) for h in range(num_heads)]
        )
        base = base * np.maximum(act_head, epsilon)
    return np.sort(np.argsort(-base)[:keep_count])


def test_cpp_decomposed_mqa_wanda_pruning_matches_python_reference_exactly():
    # True decomposed MQA (kv_num_heads == 1, more than one query head):
    # ApplyOneDecomposedGqaChain's own `is_mqa` fast path -- shared,
    # unmodified, by both the data-free and Wanda C++ entry points -- now
    # applies here too, mirroring pruning.py's own
    # `_wanda_gqa_query_head_importance`. Both ports must now agree
    # byte-for-byte.
    K, H, KVH, D, Out = 32, 8, 1, 8, 16
    model, cfg = _decomposed_gqa_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=109)

    rng = np.random.default_rng(110)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = _golden(
        _GOLDEN[
            "test_cpp_decomposed_mqa_wanda_pruning_matches_python_reference_exactly"
        ]
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    act_norm = _probe_act_norm(model, "ctx2", {"X": x_cal})
    q_keep_count = max(1, H - round(H * 0.5))
    keep_q_heads = _wanda_query_head_keep(cfg["wq"], D, H, act_norm, D, q_keep_count)
    q_idx = _head_idx(keep_q_heads, D)

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"])
    np.testing.assert_array_equal(wv_new, cfg["wv"])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])


def test_cpp_decomposed_mqa_wanda_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=1, D=8, Out=16, seed=111)
    rng = np.random.default_rng(112)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.0
    )
    assert pruned.SerializeToString() == model.SerializeToString()
