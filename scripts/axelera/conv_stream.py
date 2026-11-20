import sys, time, threading
import numpy as np
from axelera.runtime import Context
mj, n_inst, n, ops = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
ctx = Context(); m = ctx.load_model(mj)
x = [np.zeros(m.inputs()[0].shape, m.inputs()[0].dtype)]
insts, outs = [], []
for _ in range(n_inst):
    conn = ctx.device_connect(None, num_sub_devices=1)
    insts.append(conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, double_buffer=True))
    outs.append([np.zeros(t.shape, t.dtype) for t in m.outputs()])
for i, inst in enumerate(insts):
    for _ in range(20): inst.run(x, outs[i])
def worker(i):
    for _ in range(n): insts[i].run(x, outs[i])
th = [threading.Thread(target=worker, args=(i,)) for i in range(n_inst)]
t0 = time.perf_counter()
for t in th: t.start()
for t in th: t.join()
dt = time.perf_counter() - t0
calls = n_inst * n
print(f"instances={n_inst} double_buffer=True: {dt/calls*1e6:.1f} us per call amortized, {ops*calls/dt/1e12:.3f} TOPS aggregate", flush=True)
