"""Parity of onnxsim's GPTQ / AdaRound options against the real AMD Quark ONNX
package (skipped unless ``quark.onnx`` is importable).

GPTQ is deterministic, so ``quark_gptq(compensate=False)`` is compared with
Quark's ``GPTQ.fasterquant`` code for code, scale for scale. That flag exists
because Quark 0.13's error-propagation step is a no-op (it indexes the
upper-triangular Cholesky factor of ``H^-1`` by column), which
``test_quark_gptq_is_round_to_nearest`` pins so a Quark fix shows up here.
AdaRound is stochastic in Quark, so only error metrics are compared.
"""

import contextlib
import io
import tempfile
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        from quark.onnx import ModelQuantizer, QConfig
        from quark.onnx.algorithm.gptq.gptq import GPTQ, GptqProcessor
    except Exception as e:  # pragma: no cover - environment dependent
        GPTQ = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    GPTQ is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_weight_rounding as wr  # noqa: E402


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark's fine-tuning presets and ``quantized_info.csv`` write scratch files
    into the current directory; keep them out of the checkout."""
    monkeypatch.chdir(tmp_path)


class _Quiet(contextlib.AbstractContextManager):
    def __enter__(self):
        self._stack = contextlib.ExitStack()
        buf = io.StringIO()
        self._stack.enter_context(contextlib.redirect_stdout(buf))
        self._stack.enter_context(contextlib.redirect_stderr(buf))
        return buf

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def _quark_gptq(w, x, bits, group_size, block_size, act_order, per_channel, sym, mse):
    g = GPTQ(w, {})
    g.configure(bits=bits, perchannel=per_channel, sym=sym, mse=mse)
    g.add_batch(x)
    h = g.H.copy()
    with _Quiet():
        out = g.fasterquant(
            blocksize=block_size,
            percdamp=0.01,
            groupsize=group_size,
            actorder=act_order,
        )
    return h, out


# -- the numpy core, bit for bit -------------------------------------------------------


def _weights_and_acts(seed, k=64, n=24):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((k, n)) * rng.uniform(0.1, 2.0, (1, n))).astype(np.float32)
    x = rng.standard_normal((200, k)).astype(np.float32)
    x *= rng.uniform(0.2, 3.0, k).astype(np.float32)
    x[:, 5] = 0.0  # a dead input channel
    return w, x


# (bits, group_size, act_order, per_channel, sym, mse): every value of every axis
_COMBOS = [
    (8, -1, False, False, True, False),
    (8, -1, True, True, True, False),
    (8, -1, False, True, False, False),
    (8, 16, False, True, True, False),
    (4, -1, False, False, True, True),
    (4, -1, True, False, True, False),
    (4, 16, False, False, True, False),
    (4, 16, False, True, True, False),
    (4, 16, True, False, True, False),
    (4, 24, False, True, False, True),
    (3, -1, True, True, False, True),
    (3, 24, False, False, True, True),
]


@pytest.mark.parametrize("bits, group_size, act_order, per_channel, sym, mse", _COMBOS)
def test_codes_and_grid_match_quark_bit_for_bit(
    bits, group_size, act_order, per_channel, sym, mse
):
    w, x = _weights_and_acts(seed=bits * 7 + group_size)
    h, (_, q_int, scale, zero) = _quark_gptq(
        w, x, bits, group_size, 16, act_order, per_channel, sym, mse
    )
    ours = wr.quark_gptq(
        w, h, bits, group_size, 16, 0.01, act_order, per_channel, sym, mse,
        compensate=False,
    )  # fmt: skip
    np.testing.assert_array_equal(ours.q_int, q_int.astype(np.float64))
    # Quark only keeps the last group's scale / zero point
    np.testing.assert_allclose(ours.scale[-1], scale, rtol=1e-5)
    np.testing.assert_array_equal(ours.zero[-1], zero)


def test_the_hessian_matches_quarks_accumulation():
    w, x = _weights_and_acts(seed=1)
    h, _ = _quark_gptq(w, x, 8, -1, 128, False, False, True, False)
    xd = x.astype(np.float64)
    np.testing.assert_allclose(h, 2.0 / len(x) * xd.T @ xd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("group_size", [-1, 16])
def test_quark_gptq_is_round_to_nearest(group_size):
    # If this fails, Quark fixed its error propagation: drop compensate=False.
    w, x = _weights_and_acts(seed=2)
    h, (_, q_int, scale, zero) = _quark_gptq(
        w, x, 4, group_size, 16, False, True, True, False
    )
    ours = wr.quark_gptq(
        w, h, 4, group_size, 16, 0.01, False, True, True, False, compensate=True
    )
    rtn = wr.quark_gptq(
        w, h, 4, group_size, 16, 0.01, False, True, True, False, compensate=False
    )
    np.testing.assert_array_equal(rtn.q_int, q_int.astype(np.float64))
    if group_size == -1:
        np.testing.assert_array_equal(ours.scale, rtn.scale)
        err = lambda r: np.mean((x @ ((r.q_int - r.zero) * r.scale) - x @ w) ** 2)  # noqa: E731
        assert err(ours) < err(rtn)


# -- a model, end to end ---------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,32] x) => (float[N,16] y)
        {
            h = MatMul(x, w1)
            t = Relu(h)
            y = MatMul(t, w2)
        }
        """
    )
    rng = np.random.default_rng(0)
    model.graph.initializer.extend(
        numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n)
        for n, s in [("w1", (32, 32)), ("w2", (32, 16))]
    )
    data = [{"x": rng.standard_normal((16, 32)).astype(np.float32)} for _ in range(4)]
    return model, data


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)


def _quark_quantize(model, data, preset, **finetune):
    from onnxruntime.quantization import CalibrationDataReader

    class R(_Reader, CalibrationDataReader):
        pass

    d = tempfile.mkdtemp()
    onnx.save(model, d + "/m.onnx")
    cfg = QConfig.get_default_config(preset)
    cfg.global_quant_config.include_cle = False
    ff = cfg.global_quant_config.extra_options.get("FastFinetune")
    if ff is not None:
        ff["EarlyStop"] = False
        ff.update(finetune)
    with _Quiet():
        ModelQuantizer(cfg).quantize_model(d + "/m.onnx", d + "/q.onnx", R(data))
    return onnx.load(d + "/q.onnx")


def _mine_quantize(model, data, algos):
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = algos
    quantizer = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = quantizer.quantize_model(model, calibration_data_reader=_Reader(data))
    return out, quantizer


def _dequantized(model, name):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    q = next(
        v for k, v in inits.items() if k.startswith(name) and k.endswith("quantized")
    )
    return (q.astype(np.float64) - inits[name + "_zero_point"]) * inits[name + "_scale"]


@pytest.mark.parametrize("per_channel", [True, False])
def test_gptq_round_to_nearest_error_equals_quarks(mlp, per_channel):
    """Quark's GPTQ (= RTN on its grid, see above) and onnxsim's pre-GPTQ
    baseline are the same weights, so the layer errors agree; onnxsim's GPTQ
    then does better."""
    model, data = mlp
    # Quark's GPTQ only reads the first calibration batch
    q0 = _quark_quantize(model, data, "A8W8")
    with _Quiet():
        quark = GptqProcessor(
            model, q0, [data[0]], {"GPTQParams": {"PerChannel": per_channel}}
        ).apply()
    x = data[0]["x"].astype(np.float64)
    w = {
        t.name: numpy_helper.to_array(t).astype(np.float64)
        for t in model.graph.initializer
    }
    h1 = np.maximum(x @ w["w1"], 0)
    quark_err = [
        np.mean((x @ _dequantized(quark, "w1") - x @ w["w1"]) ** 2),
        np.mean((h1 @ _dequantized(quark, "w2") - h1 @ w["w2"]) ** 2),
    ]

    from onnxsim.full_qdq import quantize_full_qdq

    mine_q = quantize_full_qdq(model, calibration_data=[data[0]], per_channel=True)
    _, reports = wr.gptq_int8(
        model, mine_q, [data[0]], bits=8, per_channel=per_channel, requantize=True
    )
    assert [r.error_before for r in reports] == pytest.approx(quark_err, rel=2e-3)
    assert all(r.error_after < r.error_before for r in reports)


def _e2e_error(model, quantized, x_test):
    # No graph optimizations: ORT would otherwise fuse DQ -> MatMul -> Q into
    # integer kernels that saturate on x86 CPUs without VNNI (CI runners), which
    # makes the error depend on the host rather than on the quantization.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    feed = {"x": x_test}
    ref = ort.InferenceSession(model.SerializeToString(), so).run(None, feed)[0]
    got = ort.InferenceSession(quantized.SerializeToString(), so).run(None, feed)[0]
    return float(np.mean((got - ref) ** 2))


@pytest.mark.parametrize("drop_ratio", [1.0, 0.5, 0.0])
def test_adaround_drop_ratio_end_to_end_error_tracks_quarks(mlp, drop_ratio):
    """Quark's AdaRound is stochastic (random mini-batches), so compare the
    model-output error with a tolerance: with the same drop_ratio both end up
    within a few percent of each other, and neither strays far from plain
    round-to-nearest."""
    model, data = mlp
    x_test = np.random.default_rng(99).standard_normal((256, 32)).astype(np.float32)
    plain = _e2e_error(model, _quark_quantize(model, data, "A8W8"), x_test)
    quark = _e2e_error(
        model,
        _quark_quantize(
            model, data, "A8W8_ADAROUND", NumIterations=300, DropRatio=drop_ratio
        ),
        x_test,
    )
    mine_out, quantizer = _mine_quantize(
        model, data, [qc.AdaRoundConfig(num_iterations=300, drop_ratio=drop_ratio)]
    )
    mine = _e2e_error(model, mine_out, x_test)
    assert quantizer.last_weight_rounding["adaround"]
    assert mine == pytest.approx(quark, rel=0.15)
    assert 0.8 * plain < mine < 1.2 * plain
    assert 0.8 * plain < quark < 1.2 * plain


# -- GPTQ whatever the preset's weight dtype -----------------------------------------------

_WEIGHT_DTYPE_PRESETS = ["A8W8", "A16W8", "U8U8_AAWA", "U8S8_AAWS", "INT16_CNN_DEFAULT"]


def _quark_gptq_pipeline(model, data, preset):
    """Quark's end-to-end GPTQ run (the algorithm list of the old config API)."""
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx.quantization.config.algorithm import GPTQConfig

    class R(_Reader, CalibrationDataReader):
        pass

    d = tempfile.mkdtemp()
    onnx.save(model, d + "/m.onnx")
    cfg = QConfig.get_default_config(preset)
    cfg.global_quant_config.include_cle = False
    with _Quiet():
        ModelQuantizer(cfg).quantize_model(
            d + "/m.onnx", d + "/q.onnx", R(data), algorithms=[GPTQConfig()]
        )
    return onnx.load(d + "/q.onnx")


def _weight_grid(model, name):
    """``(codes - zero_point, scale)`` of a weight, whatever the naming scheme."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    dq = next(
        n
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0].startswith(name)
    )
    codes, scale = inits[dq.input[0]], inits[dq.input[1]]
    zp = inits[dq.input[2]] if len(dq.input) > 2 else np.zeros((), np.int64)
    return codes.astype(np.int64) - zp.astype(np.int64), np.asarray(scale, np.float64)


def test_gptq_ignores_the_presets_weight_dtype_in_quark_and_here(mlp):
    """Quark's GPTQ never raises for int16 / uint8 / asymmetric weight presets:
    it re-grids the float weights to 8 bits (a weight-only uint8 graph) whatever
    the preset says, so every preset ends up with the same weights. So does
    the compat layer (it raised for int16 weights and approximated uint8)."""
    model, data = mlp
    quark = {p: _quark_gptq_pipeline(model, data, p) for p in _WEIGHT_DTYPE_PRESETS}
    for p, q in quark.items():
        assert {
            t.data_type for t in q.graph.initializer if t.name.endswith("_quantized")
        } == {onnx.TensorProto.UINT8}, p
    mine = {}
    for p in _WEIGHT_DTYPE_PRESETS:
        cfg = qc.QConfig.get_default_config(p)
        cfg.algo_config = [qc.GPTQConfig(per_channel=False)]  # Quark's own default
        quantizer = qc.ModelQuantizer(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mine[p] = quantizer.quantize_model(
                model, calibration_data_reader=_Reader(data)
            )
        assert len(quantizer.last_weight_rounding["gptq"]) == 2, p
    ref = "A8W8"
    for name in ("w1", "w2"):
        q_ref, _ = _weight_grid(quark[ref], name)
        m_ref, _ = _weight_grid(mine[ref], name)
        for p in _WEIGHT_DTYPE_PRESETS:
            q_codes, q_scale = _weight_grid(quark[p], name)
            m_codes, m_scale = _weight_grid(mine[p], name)
            np.testing.assert_array_equal(q_codes, q_ref)  # Quark: dtype-blind
            np.testing.assert_array_equal(m_codes, m_ref)  # and so are we
            # the grid is Quark's: one per-tensor scale (repeated per column there)
            np.testing.assert_allclose(
                m_scale.reshape(-1)[0], q_scale.reshape(-1)[0], rtol=1e-6
            )
        # error propagation moves some codes, never by more than a few steps
        assert np.abs(m_ref - q_ref).max() <= 8
