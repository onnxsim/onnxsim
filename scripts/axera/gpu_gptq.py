"""GPTQ column solve on the GPU via WebGPU (wgpu-py, Vulkan backend).

Optional: needs ``pip install wgpu`` and a WebGPU adapter. ``llm_int4_requant --backend webgpu`` uses it.

Same algorithm as llm_int4_requant._gptq_codes / onnxsim._gptq_quantize_columns:
for each block of ``bw`` columns, every row is rounded column by column (one GPU thread per
row, rows are independent given the shared factor U), then the block's rounding errors are
pushed onto the not-yet-rounded columns with one GEMM. float32 on the GPU (the CPU path is
float64), so codes can differ at rounding ties; the report compares output error instead."""

import numpy as np

BW = 128  # block width (<= the thread's private array size below)
PARAM_BYTES = 32

COLUMN_WGSL = """
struct Params { rows: u32, cin: u32, i1: u32, bw: u32, lo: f32, hi: f32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read_write> W: array<f32>;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> E: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= P.rows) { return; }
    let s = S[r];
    var w1: array<f32, 128>;
    for (var j = 0u; j < P.bw; j = j + 1u) {
        w1[j] = W[r * P.cin + P.i1 + j];
    }
    for (var i = 0u; i < P.bw; i = i + 1u) {
        let col = w1[i];
        let code = clamp(round(col / s), P.lo, P.hi);
        Q[r * P.cin + P.i1 + i] = code;
        let row_u = (P.i1 + i) * P.cin + P.i1;
        let e = (col - code * s) / U[row_u + i];
        E[r * P.bw + i] = e;
        for (var j = i; j < P.bw; j = j + 1u) {
            w1[j] = w1[j] - e * U[row_u + j];
        }
    }
}
"""

GEMM_WGSL = """
struct Params { rows: u32, cin: u32, i1: u32, bw: u32, lo: f32, hi: f32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read_write> W: array<f32>;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> E: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

// W[r, c] -= sum_k E[r, k] * U[i1 + k, c] for c >= i1 + bw
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = P.i1 + P.bw + gid.x;
    let r = gid.y;
    if (r >= P.rows || c >= P.cin) { return; }
    var acc = 0.0;
    for (var k = 0u; k < P.bw; k = k + 1u) {
        acc = acc + E[r * P.bw + k] * U[(P.i1 + k) * P.cin + c];
    }
    W[r * P.cin + c] = W[r * P.cin + c] - acc;
}
"""


class GpuGptq:
    def __init__(self):
        import wgpu  # optional dependency: only the webgpu backend needs it

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync(required_limits={})
        self.info = adapter.info
        d = self.device
        self.col_mod = d.create_shader_module(code=COLUMN_WGSL)
        self.gemm_mod = d.create_shader_module(code=GEMM_WGSL)
        self.col_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self.col_mod, "entry_point": "main"}
        )
        self.gemm_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self.gemm_mod, "entry_point": "main"}
        )
        self.usage_st = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC
        )
        self.usage_un = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

    def _buf(self, nbytes, usage=None):
        return self.device.create_buffer(
            size=max(int(nbytes), 16), usage=usage or self.usage_st
        )

    def codes(self, w, u, s, lo, hi):
        """Column-sequential GPTQ codes for ``w`` [rows, cin] (any float), upper Cholesky
        factor ``u`` [cin, cin] of H^-1 (u^T u = H^-1), per-row scale ``s`` [rows].
        Returns float32 codes [rows, cin] (integers in [lo, hi])."""
        d = self.device
        rows, cin = w.shape
        w32 = np.ascontiguousarray(w, np.float32)
        u32 = np.ascontiguousarray(u, np.float32)
        s32 = np.ascontiguousarray(s, np.float32).reshape(rows)
        wb = self._buf(w32.nbytes)
        ub = self._buf(u32.nbytes)
        sb = self._buf(s32.nbytes)
        qb = self._buf(rows * cin * 4)
        eb = self._buf(rows * BW * 4)
        d.queue.write_buffer(wb, 0, w32)
        d.queue.write_buffer(ub, 0, u32)
        d.queue.write_buffer(sb, 0, s32)
        enc = d.create_command_encoder()
        for i1 in range(0, cin, BW):
            bw = min(BW, cin - i1)
            i2 = i1 + bw
            pb = self._buf(PARAM_BYTES, self.usage_un)
            d.queue.write_buffer(
                pb,
                0,
                np.array([rows, cin, i1, bw], np.uint32).tobytes()
                + np.array([lo, hi], np.float32).tobytes()
                + np.zeros(2, np.uint32).tobytes(),
            )
            bg = d.create_bind_group(
                layout=self.col_pipe.get_bind_group_layout(0),
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": wb, "offset": 0, "size": wb.size},
                    },
                    {
                        "binding": 1,
                        "resource": {"buffer": ub, "offset": 0, "size": ub.size},
                    },
                    {
                        "binding": 2,
                        "resource": {"buffer": sb, "offset": 0, "size": sb.size},
                    },
                    {
                        "binding": 3,
                        "resource": {"buffer": qb, "offset": 0, "size": qb.size},
                    },
                    {
                        "binding": 4,
                        "resource": {"buffer": eb, "offset": 0, "size": eb.size},
                    },
                    {
                        "binding": 5,
                        "resource": {"buffer": pb, "offset": 0, "size": PARAM_BYTES},
                    },
                ],
            )
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.col_pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups((rows + 63) // 64, 1, 1)
            cp.end()
            if i2 < cin:
                gb = d.create_bind_group(
                    layout=self.gemm_pipe.get_bind_group_layout(0),
                    entries=[
                        {
                            "binding": 0,
                            "resource": {"buffer": wb, "offset": 0, "size": wb.size},
                        },
                        {
                            "binding": 1,
                            "resource": {"buffer": ub, "offset": 0, "size": ub.size},
                        },
                        {
                            "binding": 2,
                            "resource": {"buffer": eb, "offset": 0, "size": eb.size},
                        },
                        {
                            "binding": 3,
                            "resource": {
                                "buffer": pb,
                                "offset": 0,
                                "size": PARAM_BYTES,
                            },
                        },
                    ],
                )
                gp = enc.begin_compute_pass()
                gp.set_pipeline(self.gemm_pipe)
                gp.set_bind_group(0, gb)
                gp.dispatch_workgroups((cin - i2 + 15) // 16, (rows + 15) // 16, 1)
                gp.end()
        d.queue.submit([enc.finish()])
        raw = d.queue.read_buffer(qb, 0, rows * cin * 4)
        return np.frombuffer(raw, np.float32).reshape(rows, cin).copy()


def upper_factor(h, damping):
    """``u`` with ``u^T u = (H + damping*mean(diag H) I)^-1``, as the requant's gptq() forms it.
    Dead diagonals must already be set to 1 by the caller."""
    h = np.asarray(h, np.float64)
    hinv = np.linalg.inv(h + damping * np.mean(np.diag(h)) * np.eye(len(h)))
    return np.linalg.cholesky((hinv + hinv.T) / 2).T
