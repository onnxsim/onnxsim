"""Parity of :mod:`onnxsim.quark_finetune` (the numpy port of Quark's
``FastFinetune`` AdaRound / AdaQuant) with the real AMD Quark ONNX package
(skipped unless ``quark.onnx`` and ``torch`` are importable).

Quark's fine-tuning is deterministic given ``FixedSeed``: its only randomness
is ``torch.randperm`` (the mini-batch of every iteration), ``torch.rand_like``
(``DropRatio`` mixing) and the weights ``torch.nn.Conv2d`` & co. draw when the
torch module of each layer is constructed. So the comparison is done twice:

* **Exact**: Quark's own ``Subgraph`` / ``convert_onnx_to_torch`` are used to
  replay that random stream into :func:`onnxsim.quark_finetune.finetune`
  (``perm_fn`` / ``rand_fn`` / ``block_hook``), after which the integer weight
  codes must agree with Quark's. Measured: AdaRound is identical on every
  probed model and option (Conv incl. grouped / dilated / asymmetric-pad /
  1-D, ConvTranspose, InstanceNormalization, LayerNormalization, MatMul on
  3-D activations, Gemm with ``alpha`` / ``beta`` / ``transB``, Relu / Clip
  following the op, ``OutputQDQ``, ``DropRatio``, ``BatchSize``,
  ``LRAdjust``, ``Parallel``, ``NumBatches`` / ``EarlyStop`` / ``WarmStart``),
  except <= 1 code in 32 on the last layer of one ConvTranspose model
  (float32 vs float64 ULPs at a rounding boundary: ``tol`` below). AdaQuant is
  identical for short runs; for long ones both Quark and this port are
  *chaotic* (its straight-through rounding flips amplify float32 vs float64
  noise: Quark vs Quark on 1e-7-perturbed data disagrees on 25-40 % of the
  codes), so those cases are compared statistically.
* **Statistical**: with the default numpy stream, layer reconstruction errors
  and the end-to-end output error land within the stated tolerances of
  Quark's.

``tests/test_quark_finetune.py`` has the Quark-free unit tests.
"""

import contextlib
import copy
import inspect
import io
import logging
import re
import tempfile
import warnings
from pathlib import Path

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
        import torch
        from quark.onnx import ModelQuantizer, QConfig
        from quark.onnx.algorithm.finetuning.fast_finetune import fast_finetune
        from quark.onnx.algorithm.finetuning.onnx_subgraph import Subgraph
        from quark.onnx.algorithm.finetuning.torch_utils import (
            convert_onnx_to_torch,
            setup_seed,
        )
        from quark.onnx.algorithm.finetuning.train_torch.train_model_loss import (
            TrainLoss,
        )
        from quark.onnx.algorithm.finetuning.train_torch.train_model_param import (
            TrainParameters,
        )
        from quark.onnx.quantization.config import algorithm as quark_algo
    except Exception as e:  # pragma: no cover - environment dependent
        torch = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    torch is None, reason="AMD Quark (amd-quark) and torch are not installed"
)

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_finetune as qf  # noqa: E402


@pytest.fixture(autouse=True)
def _scratch_dir(tmp_path, monkeypatch):
    """Quark writes scratch files (sym_shape_infer_temp.onnx, quantized_info.csv,
    ...) into the current directory."""
    monkeypatch.chdir(tmp_path)


class _Quiet(contextlib.AbstractContextManager):
    """Silences Quark and collects what it logs (to stdout when run by hand,
    through ``logging`` under pytest) into the returned buffer."""

    def __enter__(self):
        self.buf = io.StringIO()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(self.buf))
        self._stack.enter_context(contextlib.redirect_stderr(self.buf))
        # Quark's ScreenLogger loggers do not propagate and write to the stderr
        # they saw at import time, so tap them directly
        handler = logging.StreamHandler(self.buf)
        for name, lg in list(logging.root.manager.loggerDict.items()):
            if name.startswith("quark.") and isinstance(lg, logging.Logger):
                lg.addHandler(handler)
                self._stack.callback(lg.removeHandler, handler)
        return self.buf

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class _Reader:
    """Quark's CachedDataReader protocol (``get_next`` + ``reset_iter``)."""

    def __init__(self, data):
        self.data, self.i = data, 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return self.data[self.i - 1]

    def reset_iter(self):
        self.i = 0


# -- models ----------------------------------------------------------------------------
#
# Quark keys its sub-models by node name, so every node gets a distinct one (with
# empty names Quark collapses all layers into one entry and tunes only the last).


def _w(rng, name, *shape, scale=0.3):
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * scale).astype(np.float32), name
    )


def _build(kind):
    rng = np.random.default_rng(0)
    if kind == "A":  # Conv-Relu-Conv-Relu-Flatten-Gemm
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            g (float[N,3,8,8] x) => (float[N,4] y)
            {
                c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)
                r = Relu(c1)
                c2 = Conv<pads=[1,1,1,1], strides=[2,2]>(r, w2, b2)
                r2 = Relu(c2)
                f = Flatten(r2)
                y = Gemm<transB=1>(f, w3, b3)
            }
            """
        )
        inits = [
            _w(rng, "w1", 6, 3, 3, 3),
            _w(rng, "b1", 6, scale=0.1),
            _w(rng, "w2", 8, 6, 3, 3),
            _w(rng, "b2", 8, scale=0.1),
            _w(rng, "w3", 4, 128, scale=0.2),
            _w(rng, "b3", 4, scale=0.1),
        ]
        shape = (4, 3, 8, 8)
    elif kind == "B":  # Conv-InstanceNorm-LeakyRelu-ConvTranspose-Sigmoid
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            g (float[N,3,6,6] x) => (float[N,2,12,12] y)
            {
                c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)
                n1 = InstanceNormalization(c1, g1, be1)
                a = LeakyRelu<alpha=0.1>(n1)
                ct = ConvTranspose<strides=[2,2], kernel_shape=[2,2]>(a, wt, bt)
                y = Sigmoid(ct)
            }
            """
        )
        inits = [
            _w(rng, "w1", 4, 3, 3, 3),
            _w(rng, "b1", 4, scale=0.1),
            numpy_helper.from_array(
                (1 + rng.standard_normal(4) * 0.2).astype(np.float32), "g1"
            ),
            _w(rng, "be1", 4, scale=0.1),
            _w(rng, "wt", 4, 2, 2, 2),
            _w(rng, "bt", 2, scale=0.1),
        ]
        shape = (4, 3, 6, 6)
    elif kind == "C":  # LayerNorm-MatMul-Gelu-MatMul-Tanh on [N, T, D]
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 20]>
            g (float[N,5,8] x) => (float[N,5,6] y)
            {
                ln = LayerNormalization<axis=-1>(x, g1, be1)
                h = MatMul(ln, wm1)
                a = Gelu(h)
                t = MatMul(a, wm2)
                y = Tanh(t)
            }
            """
        )
        inits = [
            numpy_helper.from_array(
                (1 + rng.standard_normal(8) * 0.2).astype(np.float32), "g1"
            ),
            _w(rng, "be1", 8, scale=0.1),
            _w(rng, "wm1", 8, 12, scale=0.4),
            _w(rng, "wm2", 12, 6, scale=0.4),
        ]
        shape = (4, 5, 8)
    elif kind == "D":  # grouped / dilated / asymmetric-pad Conv, depthwise Conv
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            g (float[N,4,9,9] x) => (float[N,8,4,8] y)
            {
                c1 = Conv<group=2, dilations=[2,1], pads=[1,0,2,1], strides=[2,1]>(x, w1, b1)
                r = Relu(c1)
                y = Conv<group=8, pads=[1,1,1,1]>(r, w2, b2)
            }
            """
        )
        inits = [
            _w(rng, "w1", 8, 2, 3, 3),
            _w(rng, "b1", 8, scale=0.1),
            _w(rng, "w2", 8, 1, 3, 3),
            _w(rng, "b2", 8, scale=0.1),
        ]
        shape = (4, 4, 9, 9)
    elif (
        kind == "F"
    ):  # MatMul-Relu-MatMul on [N, K] (the model of test_quark_gptq_parity)
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 21]>
            g (float[N,32] x) => (float[N,16] y)
            {
                h = MatMul(x, w1)
                t = Relu(h)
                y = MatMul(t, w2)
            }
            """
        )
        inits = [
            numpy_helper.from_array(
                rng.standard_normal((32, 32)).astype(np.float32), "w1"
            ),
            numpy_helper.from_array(
                rng.standard_normal((32, 16)).astype(np.float32), "w2"
            ),
        ]
        shape = (16, 32)
    else:  # "E": 1-D Conv, Clip, Gemm with alpha / beta
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            g (float[N,3,10] x) => (float[N,3] y)
            {
                c1 = Conv<pads=[1,2], strides=[2]>(x, w1, b1)
                r = Clip(c1, cmin, cmax)
                f = Flatten(r)
                y = Gemm<alpha=0.5, beta=2.0>(f, w3, b3)
            }
            """
        )
        inits = [
            _w(rng, "w1", 4, 3, 3, scale=1.0),
            _w(rng, "b1", 4, scale=0.1),
            _w(rng, "w3", 24, 3),
            _w(rng, "b3", 3, scale=0.1),
            numpy_helper.from_array(np.array(0.0, np.float32), "cmin"),
            numpy_helper.from_array(np.array(1.0, np.float32), "cmax"),
        ]
        shape = (4, 3, 10)
    model.graph.initializer.extend(inits)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    data = [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(4)]
    return model, data


def _quark_quantize(model, data, preset="A8W8", **finetune):
    from onnxruntime.quantization import CalibrationDataReader

    class R(_Reader, CalibrationDataReader):
        pass

    d = tempfile.mkdtemp()
    onnx.save(model, d + "/m.onnx")
    cfg = copy.deepcopy(QConfig.get_default_config(preset))  # presets are shared
    cfg.global_quant_config.include_cle = False
    ff = cfg.global_quant_config.extra_options.get("FastFinetune")
    if ff is not None:
        # other test modules edit Quark's shared preset dicts in place: start from
        # a known state, not whatever they left
        ff.clear()
        ff.update(_ff(ff_algorithm(preset)))
        ff.update(finetune)
    with _Quiet() as buf:
        ModelQuantizer(cfg).quantize_model(d + "/m.onnx", d + "/q.onnx", R(data))
    return onnx.load(d + "/q.onnx"), buf.getvalue()


def ff_algorithm(preset):
    return "adaround" if "ADAROUND" in preset or "ACCURATE" in preset else "adaquant"


_CACHE = {}


def _prepared(kind):
    """``(float model, calibration, Quark-quantized model)``, quantized once."""
    if kind not in _CACHE:
        model, data = _build(kind)
        q, _ = _quark_quantize(model, data)
        _CACHE[kind] = (model, data, q)
    return _CACHE[kind]


def _ff(algorithm, **extra):
    ff = dict(
        DataSize=1000, FixedSeed=1705472343, BatchSize=2, NumIterations=100,
        OptimAlgorithm=algorithm, OptimDevice="cpu", InferDevice="cpu",
        EarlyStop=False, UpdateBias=False, OutputQDQ=False, DropRatio=1.0,
        LearningRate=0.1 if algorithm == "adaround" else 1e-5,
    )  # fmt: skip
    ff.update(extra)
    return ff


def _options(ff):
    return qf.FinetuneOptions(
        algorithm=ff["OptimAlgorithm"],
        num_iterations=ff["NumIterations"],
        learning_rate=ff["LearningRate"],
        batch_size=ff["BatchSize"],
        num_batches=ff.get("NumBatches", 1),
        early_stop=ff["EarlyStop"],
        update_bias=ff["UpdateBias"],
        output_qdq=ff["OutputQDQ"],
        drop_ratio=ff["DropRatio"],
        lr_adjust=ff.get("LRAdjust"),
        warm_start=ff.get("WarmStart", 0.2),
        reg_param=ff.get("RegParam", 0.01),
        beta_range=ff.get("BetaRange", (20.0, 2.0)),
        parallel=ff.get("Parallel", False),
        select_max_mem_layer=ff.get("SelectMaxMemLayer", False),
        selective_update=ff.get("SelectiveUpdate", False),
        guard=False,
    )


def _both(kind, ff, replay=True):
    """Quark's fine-tuned model and ours, same start. With ``replay`` the torch
    random stream (module construction, ``randperm``, ``rand_like``) is replayed
    into ours."""
    model, data, q = _prepared(kind)
    with _Quiet() as buf:
        quark_out = fast_finetune(model, q, False, _Reader(data), {"FastFinetune": ff})
    log = buf.getvalue()
    kwargs = {}
    if replay:
        with _Quiet():
            sg = Subgraph(model, q, False, _Reader(data), {"FastFinetune": ff})
            setup_seed(ff["FixedSeed"])

        def hook(layer, _name):
            with _Quiet():
                bias = sg.f_bias_list[layer]
                convert_onnx_to_torch(
                    sg.subgraph_qmodel_list[layer],
                    np.array(sg.f_weight_list[layer]),
                    None if bias is None else np.array(bias).reshape(-1),
                )

        kwargs = dict(
            perm_fn=lambda n: torch.randperm(n).numpy(),
            rand_fn=lambda shape: torch.rand(shape).numpy(),
            block_hook=hook,
        )
    trace = []
    mine, reports = qf.finetune(model, q, data, _options(ff), trace=trace, **kwargs)
    return quark_out, mine, q, reports, trace, log


def _codes(model):
    return {
        t.name: numpy_helper.to_array(t).astype(np.int64)
        for t in model.graph.initializer
        if t.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.INT32)
        and numpy_helper.to_array(t).ndim > 0
    }


def _mismatch(a, b):
    ca, cb = _codes(a), _codes(b)
    return {k: float(np.mean(ca[k] != cb[k])) for k in ca}


# -- exact, given the same mini-batches (AdaRound) -------------------------------------------

_ADAROUND_CASES = [
    # (model, FastFinetune overrides, tolerance on the fraction of differing codes)
    ("A", {}, 0.0),
    ("A", dict(OutputQDQ=True, BatchSize=3), 0.0),
    ("A", dict(DropRatio=0.5, BatchSize=4), 0.0),
    ("A", dict(DropRatio=0.0, BatchSize=8), 0.0),
    ("A", dict(LRAdjust=(0.0001, 0.02), Parallel=True), 0.0),
    ("A", dict(EarlyStop=True, NumBatches=7, NumIterations=150, WarmStart=0.1, BatchSize=1), 0.0),
    ("A", dict(RegParam=0.05, BetaRange=(10, 1)), 0.0),
    ("B", {}, 0.0),
    ("B", dict(OutputQDQ=True, BatchSize=3), 0.04),
    ("B", dict(DropRatio=0.5, BatchSize=4), 0.04),
    ("C", {}, 0.0),
    ("C", dict(OutputQDQ=True, BatchSize=3), 0.0),
    ("C", dict(DropRatio=0.5, BatchSize=4), 0.0),
    ("D", {}, 0.0),
    ("D", dict(OutputQDQ=True, BatchSize=3), 0.0),
    ("E", {}, 0.0),
    ("E", dict(OutputQDQ=True, BatchSize=3), 0.0),
    ("F", {}, 0.0),
    ("F", dict(OutputQDQ=True, BatchSize=4, DropRatio=0.5), 0.0),
]  # fmt: skip


@pytest.mark.parametrize("kind, extra, tol", _ADAROUND_CASES)
def test_adaround_codes_equal_quarks_given_the_same_minibatches(kind, extra, tol):
    ff = _ff("adaround", **extra)
    quark_out, mine, q, _, trace, _ = _both(kind, ff)
    changed = _mismatch(q, quark_out)
    assert any(v > 0 for v in changed.values())  # Quark really tuned something
    diff = _mismatch(quark_out, mine)
    assert max(diff.values()) <= tol, diff
    if tol == 0.0:
        assert diff == {k: 0.0 for k in diff}


def test_adaround_loss_trace_matches_quarks_log():
    quark_out, mine, q, _, trace, log = _both("A", _ff("adaround"))
    rows = re.findall(
        r"adaround iterations=(\d+), lr=[\d.e-]+, loss=[\d.]+ "
        r"\(Recons loss=([\d.]+), Rounding loss=([\d.]+)\)",
        log,
    )
    layers, cur = [], []
    for it, rec, rnd in rows:
        if int(it) == 0 and cur:
            layers.append(cur)
            cur = []
        cur.append((int(it), float(rec), float(rnd)))
    layers.append(cur)
    assert len(layers) == len(trace) == 3
    for ql, tl in zip(layers, trace):
        for it, rec, rnd in ql:
            assert tl[it][1] == pytest.approx(rec, abs=2e-6, rel=2e-2)
            assert tl[it][2] == pytest.approx(rnd, abs=2e-2)


def test_the_blocks_are_the_ones_quark_trains():
    model, data, q = _prepared("B")
    with _Quiet():
        sg = Subgraph(model, q, False, _Reader(data), {"FastFinetune": _ff("adaround")})
    ours = qf._find_blocks(model, q, qf.FinetuneOptions())
    assert [b.op_type for b in ours] == [
        next(n.op_type for n in m.graph.node if n.name == name)
        for name, m in sg.subgraph_qmodel.items()
    ]
    assert [b.f_end for b in ours] == [
        out for _, out in sg.fsubgraph_input_output_tensors.values()
    ]


# -- AdaQuant -----------------------------------------------------------------------------------

_ADAQUANT_EXACT = [
    ("A", dict(UpdateBias=True, LearningRate=1e-3, NumIterations=30, BatchSize=2)),
    ("A", dict(UpdateBias=True, LearningRate=1e-4, NumIterations=20, BatchSize=4)),
    ("E", dict(UpdateBias=True, LearningRate=1e-3, NumIterations=30, BatchSize=4)),
    ("F", dict(LearningRate=1e-3, NumIterations=30, BatchSize=4)),
]


@pytest.mark.parametrize("kind, extra", _ADAQUANT_EXACT)
def test_adaquant_codes_equal_quarks_for_short_runs(kind, extra):
    quark_out, mine, q, _, _, _ = _both(kind, _ff("adaquant", **extra))
    changed = _mismatch(q, quark_out)
    assert any(v > 0 for v in changed.values())
    assert max(_mismatch(quark_out, mine).values()) == 0.0


@pytest.mark.parametrize("iterations, num_batches", [(40, 1), (50, 1), (50, 5)])
def test_adaquant_early_stop_and_update_bias_follow_quark(iterations, num_batches):
    """Quark's early-stop rule fires at the same iteration of every layer
    (window ``num_iterations / 10`` or ``num_batches``) and the resulting
    weight and bias codes are the same."""
    ff = _ff(
        "adaquant", EarlyStop=True, UpdateBias=True, LearningRate=1e-3,
        NumIterations=iterations, BatchSize=2, NumBatches=num_batches,
    )  # fmt: skip
    quark_out, mine, q, _, trace, log = _both("A", ff)
    # per layer: the iteration Quark broke at (its log line), or None
    stops = []
    for seg in re.split(r"will be optimized by", log)[1:]:
        m = re.search(r"Iterations=(\d+), mean loss", seg)
        stops.append(int(m.group(1)) if m else None)
    ours = [len(t) - 1 if len(t) < iterations else None for t in trace]
    assert any(stops)  # Quark stopped early somewhere
    assert ours == stops
    assert max(_mismatch(quark_out, mine).values()) == 0.0
    assert changed_bias(q, quark_out) and changed_bias(q, mine)


def changed_bias(a, b):
    ca, cb = _codes(a), _codes(b)
    return any(not np.array_equal(ca[k], cb[k]) for k in ca if k.startswith("b"))


def test_long_adaquant_runs_are_chaotic_in_quark_itself():
    """200 iterations at lr 1e-4: even Quark disagrees with itself on a 1e-7
    perturbation of the data, about as much as it disagrees with this port."""
    kind, extra = "A", dict(LearningRate=1e-4, NumIterations=200, BatchSize=4)
    ff = _ff("adaquant", **extra)
    model, data, q = _prepared(kind)
    quark_out, mine, *_ = _both(kind, ff)
    prng = np.random.default_rng(5)
    data2 = [
        {
            "x": (d["x"] * (1 + 1e-7 * prng.standard_normal(d["x"].shape))).astype(
                np.float32
            )
        }
        for d in data
    ]
    with _Quiet():
        perturbed = fast_finetune(model, q, False, _Reader(data2), {"FastFinetune": ff})
    d_self = np.mean(list(_mismatch(quark_out, perturbed).values()))
    d_mine = np.mean(list(_mismatch(quark_out, mine).values()))
    assert d_self > 0.1  # chaotic
    assert d_mine <= d_self + 0.2


# -- losses -------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 7), (3, 4, 5, 5), (3, 6, 4)])
def test_reconstruction_loss_equals_quarks(shape):
    r = np.random.default_rng(1)
    a, b = (
        r.standard_normal(shape).astype(np.float32),
        r.standard_normal(shape).astype(np.float32),
    )
    quark = float(TrainLoss.calc_recon_loss(torch.from_numpy(a), torch.from_numpy(b)))
    blk = qf._Block(
        "t", "Gemm", qf._MatMulOp(False), np.eye(2), None, None, None, None, 1.0, 1.0,
        None, "", "", "", None, None,
    )  # type: ignore[arg-type]  # fmt: skip
    mine = qf._recon_grad(
        blk, (a, a, a, None), a.astype(np.float64), b.astype(np.float64)
    )[0]
    assert mine == pytest.approx(quark, rel=1e-5)


def test_round_loss_and_beta_schedule_equal_quarks():
    r = np.random.default_rng(2)
    alpha = r.standard_normal((6, 9)).astype(np.float32) * 3
    params = TrainParameters(
        num_iterations=100, reg_param=0.03, beta_range=(18, 3), warm_start=0.25
    )
    for it in (0, 24, 25, 40, 70, 99):
        quark = float(TrainLoss.calc_round_loss(torch.from_numpy(alpha), params, it))
        if it < 25:
            mine = 0.0
        else:
            sig = 1 / (1 + np.exp(-alpha.astype(np.float64)))
            h = np.clip(sig * 1.2 - 0.1, 0, 1)
            beta = qf._beta(100, it, (18, 3), 0.25)
            mine = 0.03 * float(np.sum(1 - np.abs(2 * h - 1) ** beta))
            assert beta == pytest.approx(
                float(TrainLoss._calculate_beta(100, it, (18, 3), 0.25))
            )
        assert mine == pytest.approx(quark, rel=1e-4, abs=1e-6)


# -- statistical agreement with the default numpy stream ----------------------------------------


def _e2e(model, quantized, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(model.SerializeToString(), so).run(None, {"x": x})[0]
    got = ort.InferenceSession(quantized.SerializeToString(), so).run(None, {"x": x})[0]
    return float(np.mean((got - ref) ** 2))


def _mine_quantize(model, data, algo, **params):
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [algo(guard=False, **params)]
    quantizer = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = quantizer.quantize_model(model, calibration_data_reader=_Reader(data))
    return out, quantizer


def _eight_batches(kind="A"):
    model, _ = _build(kind)
    rng = np.random.default_rng(3)
    return model, [
        {"x": rng.standard_normal((4, 3, 8, 8)).astype(np.float32)} for _ in range(8)
    ]


@pytest.mark.parametrize("drop_ratio", [1.0, 0.5, 0.0])
def test_adaround_end_to_end_error_tracks_quarks(drop_ratio):
    """Both optimize 300 iterations over mini-batches of 2: the output error
    of the quantized CNN is within 15 % of Quark's, and neither is far from
    plain round-to-nearest (the model is small, so the gain is a few percent
    -- the tolerance is the stochastic spread of both implementations)."""
    model, data = _eight_batches()
    x_test = (
        np.random.default_rng(99).standard_normal((256, 3, 8, 8)).astype(np.float32)
    )
    plain = _e2e(model, _quark_quantize(model, data)[0], x_test)
    quark = _e2e(
        model,
        _quark_quantize(
            model,
            data,
            "A8W8_ADAROUND",
            NumIterations=300,
            DropRatio=drop_ratio,
            BatchSize=2,
        )[0],
        x_test,
    )
    out, q = _mine_quantize(
        model,
        data,
        qc.AdaRoundConfig,
        num_iterations=300,
        drop_ratio=drop_ratio,
        batch_size=2,
    )
    mine = _e2e(model, out, x_test)
    assert q.last_weight_rounding["adaround"]
    assert mine == pytest.approx(quark, rel=0.15)
    assert 0.85 * plain < mine < 1.15 * plain
    assert 0.85 * plain < quark < 1.15 * plain


@pytest.mark.parametrize("update_bias", [False, True])
def test_adaquant_end_to_end_error_tracks_quarks(update_bias):
    model, data = _eight_batches()
    x_test = (
        np.random.default_rng(99).standard_normal((256, 3, 8, 8)).astype(np.float32)
    )
    quark = _e2e(
        model,
        _quark_quantize(
            model, data, "A8W8_ADAQUANT", NumIterations=300, LearningRate=1e-5,
            UpdateBias=update_bias, BatchSize=4,
        )[0],
        x_test,
    )  # fmt: skip
    out, q = _mine_quantize(
        model, data, qc.AdaQuantConfig, num_iterations=300, learning_rate=1e-5,
        update_bias=update_bias, batch_size=4,
    )  # fmt: skip
    mine = _e2e(model, out, x_test)
    assert mine == pytest.approx(quark, rel=0.15)


@pytest.mark.parametrize("lr", [1e-5, 1e-3])
def test_adaquant_end_to_end_error_tracks_quarks_on_an_mlp(lr):
    """The onnxsim AdaQuant of old (``legacy_engine``) is another algorithm and
    lands elsewhere; this port tracks Quark's, also where Quark's own
    straight-through training makes the model *worse* than round-to-nearest
    (lr 1e-3)."""
    model, _ = _build("F")
    rng = np.random.default_rng(3)
    data = [{"x": rng.standard_normal((16, 32)).astype(np.float32)} for _ in range(8)]
    x_test = np.random.default_rng(99).standard_normal((256, 32)).astype(np.float32)
    quark = _e2e(
        model,
        _quark_quantize(
            model, data, "A8W8_ADAQUANT", NumIterations=500, LearningRate=lr,
            UpdateBias=False, BatchSize=4,
        )[0],
        x_test,
    )  # fmt: skip
    out, _ = _mine_quantize(
        model,
        data,
        qc.AdaQuantConfig,
        num_iterations=500,
        learning_rate=lr,
        batch_size=4,
    )
    assert _e2e(model, out, x_test) == pytest.approx(quark, rel=0.1)


def test_layer_reconstruction_errors_track_quarks():
    """Quark logs every layer's reconstruction error before / after tuning
    (hard-rounded weights, MSE over all samples of the quantized-input block);
    the report of this port is the same metric. Layers 0-1 land within 15 %;
    the last one trains on the previous layers' (noisy) output and is only
    required to improve."""
    model, data = _eight_batches()
    for bs in (1, 2, 4):
        _, log = _quark_quantize(
            model, data, "A8W8_ADAROUND", NumIterations=300, BatchSize=bs
        )
        quark = [
            (float(b), float(a))
            for b, a in re.findall(
                r"recons metrics was optimized from ([\d.]+) to ([\d.]+)", log
            )
        ]
        _, q = _mine_quantize(
            model, data, qc.AdaRoundConfig, num_iterations=300, batch_size=bs
        )
        mine = [
            (r.error_before, r.error_after) for r in q.last_weight_rounding["adaround"]
        ]
        assert len(quark) == len(mine) == 3
        for (qb, qa), (mb, ma) in zip(quark[:2], mine[:2]):
            assert mb == pytest.approx(qb, rel=0.1)  # the same starting point
            assert ma == pytest.approx(qa, rel=0.15)
        assert all(a <= b for b, a in quark) and all(a <= b for b, a in mine)


# -- options ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["A", "B", "C", "D", "E"])
def test_select_max_mem_layer_picks_the_layer_quark_picks(kind):
    ff = _ff("adaround", SelectMaxMemLayer=True, NumIterations=60)
    quark_out, mine, q, reports, _, _ = _both(kind, ff, replay=False)
    assert len(reports) == 1

    def trained(m):
        return {
            k for k, v in _mismatch(q, m).items() if v > 0 and not k.startswith("b")
        }

    assert trained(quark_out) and trained(mine) == trained(quark_out)


def _quark_keys():
    """Every ``FastFinetune`` key the finetuning package reads."""
    root = Path(inspect.getfile(Subgraph)).parent
    keys = set()
    for path in list(root.glob("*.py")) + list((root / "train_torch").glob("*.py")):
        keys |= set(re.findall(r'FastFinetune"\]?\)?\.get\("(\w+)"', path.read_text()))
        keys |= set(
            re.findall(
                r'"(\w+)" (?:not )?in extra_options\["FastFinetune"\]', path.read_text()
            )
        )
        keys |= set(
            re.findall(r'get\("FastFinetune", \{\}\)\.get\("(\w+)"', path.read_text())
        )
    return keys


# keys with no effect on the numbers (devices, data loading, caching, logging);
# MemOptLevel 2 / NumWorkers / DynamicBatch do change what Quark computes and are
# modelled (tests/test_quark_finetune_coverage_parity.py)
_NO_EFFECT = {
    "OptimAlgorithm", "OptimDevice", "InferDevice", "PinMemory", "LogPeriod", "UseGDS",
}  # fmt: skip


def test_every_fastfinetune_key_quark_reads_is_implemented_or_known_to_be_inert():
    keys = _quark_keys()
    assert {"BatchSize", "NumBatches", "EarlyStop", "OutputQDQ", "DropRatio"} <= keys
    handled = set(qc._FASTFT_KEYS) | _NO_EFFECT | {"SaveAndRestore", "TmpDir"}
    assert {"MemOptLevel", "NumWorkers", "DynamicBatch"} <= set(qc._FASTFT_KEYS)
    assert keys <= handled, sorted(keys - handled)
    # the keys listed as inert really are read by Quark only for non-numeric reasons
    assert _NO_EFFECT <= keys | {"OptimAlgorithm"}


@pytest.mark.parametrize(
    "cfg_cls, name", [("AdaRoundConfig", "adaround"), ("AdaQuantConfig", "adaquant")]
)
def test_algo_config_fields_defaults_and_forwarding_match_quark(cfg_cls, name):
    quark_cls = getattr(quark_algo, cfg_cls)
    sig = inspect.signature(quark_cls.__init__).parameters
    forwarded = {
        re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower().replace("l_r_adjust", "lr_adjust")
        .replace("output_q_d_q", "output_qdq").replace("use_g_d_s", "use_gds")
        for k in quark_cls()._get_config({})["FastFinetune"]
    }  # fmt: skip
    # fields Quark stores but never copies into the FastFinetune dict
    assert set(qc._FASTFT_NOT_FORWARDED) == {
        p for p in sig if p != "self" and p not in forwarded
    }
    # numeric defaults of the fields this port acts on
    mine = {
        "adaround": qf.FinetuneOptions(),
        "adaquant": qf.FinetuneOptions(algorithm="adaquant"),
    }[name]
    q = quark_cls()
    assert mine.lr() == q.learning_rate
    for attr in ("batch_size", "num_batches", "early_stop", "drop_ratio", "reg_param", "warm_start",
                 "selective_update", "output_qdq", "update_bias", "mem_opt_level", "parallel",
                 "select_max_mem_layer", "num_workers", "dynamic_batch"):  # fmt: skip
        assert getattr(mine, attr) == getattr(q, attr), attr
    assert tuple(mine.beta_range) == tuple(q.beta_range)
    assert tuple(mine.target_ops) == tuple(q.target_op_type) or set(
        mine.target_ops
    ) == set(q.target_op_type)
    assert mine.seed == q.fixed_seed
    assert set(sig) - {"self"} <= set(qc._FASTFT_KEYS.values()) | {
        "optim_device", "infer_device", "cache_dir", "log_period", "dynamic_batch",
        "num_workers", "pin_memory", "use_gds",
    }  # fmt: skip


@pytest.mark.parametrize(
    "preset", ["A8W8_ADAROUND", "A8W8_ADAQUANT", "INT8_CNN_ACCURATE", "XINT8_ADAQUANT"]
)
def test_presets_carry_the_fastfinetune_dict_of_quarks_presets(preset):
    with _Quiet():
        quark_ff = QConfig.get_default_config(preset).global_quant_config.extra_options[
            "FastFinetune"
        ]
    (algo,) = qc.QConfig.get_default_config(preset).algo_config
    p = algo.params
    assert (
        p["data_size"] == quark_ff["DataSize"]
        and p["fixed_seed"] == quark_ff["FixedSeed"]
    )
    # (other test modules edit Quark's shared preset dicts in place, so only the
    # keys nobody overrides are compared)
    assert p["learning_rate"] == quark_ff["LearningRate"]
    assert algo.name == quark_ff["OptimAlgorithm"]
