"""Check partition concurrency, argument isolation and runtime restoration."""

import importlib.util
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location(
    "cpu_thread_pool",
    Path(__file__).resolve().parents[1] / "scripts/tinygrad/cpu_thread_pool.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
cpu_thread_pool = _module.cpu_thread_pool


@unittest.skipUnless(importlib.util.find_spec("tinygrad"), "tinygrad is not installed")
class TestCPUThreadPool(unittest.TestCase):
    def setUp(self):
        from tinygrad.runtime.ops_cpu import CPUProgram

        self.program_type = CPUProgram
        self.env = patch.dict("os.environ", {"HCQ2": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_partitions_overlap_with_separate_arguments_and_complete(self):
        barrier = threading.Barrier(2, timeout=3)
        completed = []

        def native(pointer, core):
            barrier.wait()
            completed.append((pointer.value, core.value))

        program = SimpleNamespace(lvp=False, runtimevars={"core_id": 1}, fxn=native)
        original = self.program_type.__call__
        with cpu_thread_pool(2):
            elapsed = self.program_type.__call__(
                program,
                SimpleNamespace(va_addr=123),
                vals=(0,),
                global_size=(2, 1, 1),
                wait=True,
            )
            self.assertEqual(sorted(completed), [(123, 0), (123, 1)])
            self.assertGreater(elapsed, 0)
        self.assertIs(self.program_type.__call__, original)

    def test_exception_propagates_and_restores_runtime(self):
        def fail(*args):
            raise RuntimeError("kernel failure")

        program = SimpleNamespace(lvp=False, runtimevars={"core_id": 0}, fxn=fail)
        original = self.program_type.__call__
        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            with cpu_thread_pool(2):
                self.program_type.__call__(program, vals=(0,), global_size=(2, 1, 1))
        self.assertIs(self.program_type.__call__, original)

    def test_lvp_and_single_partition_keep_original_runtime(self):
        for lvp, runtimevars, size in [
            (True, {"core_id": 0}, 2),
            (False, {}, 2),
            (False, {"core_id": 0}, 1),
        ]:
            with self.subTest(lvp=lvp, runtimevars=runtimevars, size=size):
                with patch.object(
                    self.program_type, "__call__", return_value="original"
                ) as original:
                    with cpu_thread_pool(2):
                        program = SimpleNamespace(lvp=lvp, runtimevars=runtimevars)
                        self.assertEqual(
                            self.program_type.__call__(
                                program, global_size=(size, 1, 1)
                            ),
                            "original",
                        )
                        original.assert_called_once()

    def test_rejects_hcq2(self):
        with patch.dict("os.environ", {"HCQ2": "1"}):
            with self.assertRaisesRegex(ValueError, "HCQ2=0"):
                with cpu_thread_pool(2):
                    self.fail("must reject before entering")

    @unittest.skipUnless(shutil.which("clang"), "native CPU test needs clang")
    def test_native_partitioned_matvec_matches_serial_for_distinct_inputs(self):
        import numpy as np
        from tinygrad import Tensor, TinyJit
        from tinygrad.helpers import Context

        with Context(DEV="CPU", NUM_CPU_THREADS=4, BEAM=0):
            weight = Tensor(
                (np.arange(512 * 2048).reshape(512, 2048) % 17).astype(np.float32)
                * 0.002
            ).realize()
            base = np.arange(2048, dtype=np.float32).reshape(1, 2048) * 0.001
            inputs = [Tensor(value).realize() for value in (base, base * 0.5, -base)]
            x = inputs[0]

            @TinyJit
            def step(value):
                return value.linear(weight.T).realize()

            for _ in range(3):
                step(x)
            expected = [step(value).numpy() for value in inputs]
            with cpu_thread_pool(4):
                for value, want in zip(inputs, expected):
                    np.testing.assert_array_equal(step(value).numpy(), want)


if __name__ == "__main__":
    unittest.main()
