"""Opt-in parallel dispatch for tinygrad 0.14's synchronous CPU runtime.

The stock runtime invokes each generated ``core_id`` partition serially.
This context executes those independent partitions in persistent Python
threads; ctypes releases the GIL while the native kernel runs. It targets
the synchronous CPU runtime, not HCQ2 or LVP. Install once around execution,
before tuning, so search measures the runtime that will execute the model.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import os
import time


@contextmanager
def cpu_thread_pool(workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if os.environ.get("HCQ2", "0") not in ("", "0"):
        raise ValueError("cpu_thread_pool requires the synchronous runtime (HCQ2=0)")
    from tinygrad.runtime.ops_cpu import CPUProgram, MAX_ARGS

    original = CPUProgram.__call__
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tinygrad-cpu") as pool:
        def run(program, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1),
                vals=(), wait=False, timeout=None):
            if program.lvp or "core_id" not in program.runtimevars or global_size[0] < 2:
                return original(program, *bufs, global_size=global_size, local_size=local_size,
                                vals=vals, wait=wait, timeout=timeout)
            start = time.perf_counter()
            args = [*[buf.va_addr for buf in bufs], *vals]
            if len(args) > MAX_ARGS:
                raise ValueError(f"CPU programs support at most {MAX_ARGS} arguments")
            core_index = program.runtimevars["core_id"]

            def invoke(index):
                # Each task owns its argument array; only output partitions are shared.
                task_args = args.copy()
                task_args[core_index] = index
                program.fxn(*[ctypes.c_uint64(value) for value in task_args])

            list(pool.map(invoke, range(global_size[0])))
            return time.perf_counter() - start if wait else None

        CPUProgram.__call__ = run
        try:
            yield
        finally:
            CPUProgram.__call__ = original
