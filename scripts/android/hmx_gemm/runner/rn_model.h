/* The DSP graph runner's model: a static list of ops over quantized tensors, planned on the host (hmx_runner_client.c,
 * from qdq_graph.py's program) and executed on the DSP in one FastRPC call (rn_exec.h). Shared by both sides, so
 * plain fixed-size structs; everything the DSP needs besides the packed weights/params blob is in rn_model_t.
 *
 * Tensors are flat padded crouton buffers (hmx_qconv3.h qc_geom2) in VTCM at planned offsets (the graph input may
 * instead live in DDR: the RPC buffer). Ops:
 *  RN_CONV    k x k (k odd), stride 1/2, pad k/2: HMX :single taps over "sources" (the input itself, its column-shifted
 *             copies, or its stride-2 phases and their shifted copies), requantized QC_FAST / QC_EXACT.
 *  RN_ADD     ORT QLinearAdd semantics, HVX fixed point + a scalar fp32 fix for the lanes near a .5 boundary.
 *  RN_MAXPOOL 3x3 stride 2 pad 1 (input zero point 0: padding = the minimum), HVX over the same phase sources. */
#ifndef RN_MODEL_H
#define RN_MODEL_H
#include <stdint.h>

#include "../hmx_qconv3.h"

#define RN_MAX_T 64
#define RN_MAX_OPS 64
#define RN_MAX_SRC 16

enum { RN_CONV = 1, RN_ADD = 2, RN_MAXPOOL = 3 };
enum { RN_SRC_PHASE = 0, RN_SRC_SHIFT = 1 };

typedef struct {
  int c, cp, h, w, zp;  /* channels, padded to a multiple of 32 */
  float scale;
  qc_geom_t g;
  uint32_t off;         /* VTCM offset */
  int in_ddr;           /* the graph input: read from the RPC buffer, never written */
} rn_tensor_t;

typedef struct {
  int kind;             /* RN_SRC_PHASE: phase a (2*py + px) of the input; RN_SRC_SHIFT: source a (-1 = the input
                         * tensor) shifted by c columns */
  int a, c;
  uint32_t off;         /* VTCM offset, geometry = the op's gsrc */
} rn_src_t;

typedef struct {
  int type, x, x2, y;   /* tensors (x2: the Add's second input) */
  int k, s, n;          /* conv: kernel, stride, output channels */
  uint32_t w_off, w_len;     /* packed weights in the blob */
  uint32_t prm_off, prm_len; /* qc_blk_t[n/32] + qc_hdr_t in the blob */
  uint32_t vw, vprm, vplanes, vside; /* op scratch in VTCM */
  int prefetch_next;    /* conv: the next conv op, whose weights may be copied while this op (and those up to it) run */
  int prefetched;       /* conv: weights were set up by the previous conv's prefetch (else copied synchronously) */
  int side_cap;
  int nsrc;
  rn_src_t src[RN_MAX_SRC];
  qc_geom_t gsrc;       /* geometry of the sources = the output's */
  int ntaps;
  int8_t tap_src[QC_MAX_TAPS]; /* index into src, -1 = the input tensor */
  int8_t tap_drow[QC_MAX_TAPS];
  /* RN_ADD */
  float ra, rb, fixed;  /* ORT's fp32 constants */
  /* fixed point v * 2^F = sh(a * ma_hi, 12 - sa) + sh(a * ma_lo, -sa) + (same for b) + fq, where ma = ma_hi * 2^12 +
   * ma_lo is ra's 24-bit mantissa (a * ra exact) and sh(x, n) = n >= 0 ? x << n : x >> -n */
  int32_t a_hi, a_lo, b_hi, b_lo, sa, sb, fq, F, win;
} rn_op_t;

typedef struct {
  int nt, nops, input, output;
  uint32_t vtcm_bytes, blob_bytes;
  rn_tensor_t t[RN_MAX_T];
  rn_op_t op[RN_MAX_OPS];
} rn_model_t;
#endif
