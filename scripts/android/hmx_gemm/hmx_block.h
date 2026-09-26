/* Minimal HMX "block" primitives for Hexagon V69 -- what a code generator needs to emit.
 *
 * A block = one set of load pairs over K, accumulating, then one store with conversion:
 *   hmx_blk_set_table(T);                 // once per output-column block (scale/bias per column)
 *   hmx_blk_mac_f16(A, W, ktiles);        // or hmx_blk_mac_u8s8; call again to keep accumulating
 *   hmx_blk_store_f16(C);                 // or _u16 / _u8: writes one 32x32 tile, clears the accumulator
 *
 * Preconditions (the caller's runtime, see hmx_runtime.h -- keep it out of the kernel):
 *   - this thread holds the HVX lock and HAP_compute_res_hmx_lock for a compute_res context that was
 *     acquired with the HMX parameter, after a HAP_power_set_HMX power_up vote (without the vote the
 *     first HMX op wedges the cDSP until a phone reboot);
 *   - A, W, C and T are in that context's VTCM (HMX only addresses VTCM), each tile 2048-byte aligned,
 *     T (256 B) 128-byte aligned;
 *   - an operand span (ktiles * 2048 bytes, activation-side) must lie inside one 256 KB-aligned VTCM
 *     window -- a span crossing a 256 KB boundary page-faults the user PD on the phone (hexagon-sim does
 *     not model this);
 *   - fp16: at most 32 K-tiles per load pair (K <= 1024); int8: at most 8 (checked exact to K = 256;
 *     K = 1024 in one pair is wrong). hmx_blk_mac_* split longer K into pairs, which keep accumulating.
 *
 * Layouts: 32x32 tiles, IDX(i, j) = 64*(i/2) + 2*j + i%2 (row pairs, one 128-byte vector per row pair).
 *   fp16: A(r, k), W(k, c), C(r, c) are all halfword IDX(.,.); consecutive K blocks = consecutive tiles.
 *   int8: A(r, k) = byte 2*IDX(r, k) + 1 (odd bytes; even bytes ignored), 2048 B per K=32 block;
 *         W(k, c) = byte 128*(k/4) + 4*c + k%4, 1024 B per K=32 block (same Rt as A: W reads Rt/2 bytes);
 *         C via _u16: u16 IDX(r, c); via _u8: byte 2*IDX(r, c) + 1 (= the int8 activation layout).
 * Column table T (words 0..31 = output columns 0..31; words 32..63 unused):
 *   fp16 out: high 16 bits = fp16 bias added exactly to the exact accumulator, low 16 bits = 0;
 *             output = round-to-nearest-even(acc + bias) to fp16.
 *   int8 out: low 16 bits = fp16 scale s; _u16 = min(65535, floor(acc * s / 2)), _u8 = min(255,
 *             floor(acc * s / 512)); acc < 0 -> 0 (no signed int8 output, no int8 -> fp16 output on v69).
 */
#ifndef HMX_BLOCK_H
#define HMX_BLOCK_H
#include <stddef.h>
#include <stdint.h>

#define HMX_TILE 32
#define HMX_TILE_BYTES 2048
#define HMX_IDX(i, j) (64 * ((i) / 2) + 2 * (j) + ((i) % 2))
#define HMX_MAX_KTILES 32   /* fp16 load pair */
#define HMX_MAX_KTILES_I8 8 /* int8 load pair */

#ifdef __hexagon__
static inline void hmx_blk_set_table(const void* t) { __asm__ volatile("bias = mxmem(%0)" ::"r"(t) : "memory"); }
/* 64-bit column table (256 B: words 0..31 = low words, 32..63 = high words). For int8 outputs the high word is an
 * int32 added exactly to the accumulator and low-word bit 22 adds 0.5 before the floor (hexagon-sim). */
static inline void hmx_blk_set_table2(const void* t) { __asm__ volatile("bias = mxmem2(%0)" ::"r"(t) : "memory"); }

static inline void hmx_blk_mac_f16(const void* a, const void* w, int ktiles) {
  const uint8_t *pa = (const uint8_t*)a, *pw = (const uint8_t*)w;
  for (int k0 = 0; k0 < ktiles; k0 += HMX_MAX_KTILES) {
    int n = ktiles - k0 < HMX_MAX_KTILES ? ktiles - k0 : HMX_MAX_KTILES, lim = n * HMX_TILE_BYTES - 1;
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" ::"r"(pa + (size_t)k0 * HMX_TILE_BYTES),
                     "r"(lim), "r"(pw + (size_t)k0 * HMX_TILE_BYTES), "r"(lim)
                     : "memory");
  }
}

/* a: ktiles x 2048 B activation blocks, w: ktiles x 1024 B weight blocks */
static inline void hmx_blk_mac_u8s8(const void* a, const void* w, int ktiles) {
  const uint8_t *pa = (const uint8_t*)a, *pw = (const uint8_t*)w;
  for (int k0 = 0; k0 < ktiles; k0 += HMX_MAX_KTILES_I8) {
    int n = ktiles - k0 < HMX_MAX_KTILES_I8 ? ktiles - k0 : HMX_MAX_KTILES_I8, lim = n * HMX_TILE_BYTES - 1;
    __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }" ::"r"(pa + (size_t)k0 * HMX_TILE_BYTES),
                     "r"(lim), "r"(pw + (size_t)k0 * HMX_TILE_BYTES / 2), "r"(lim)
                     : "memory");
  }
}

/* int8 "cm" (channel-major) path -- what QNN's own int8 convs use (hmx_convbbb1x1_stride1 in its V69 skel):
 * one instruction = 64 spatial rows x 32 input channels x 32 (or 64 with weight :deep) output channels, i.e.
 * 2x the rows of the fp16 / non-cm int8 path for the same 2 KB activation crouton, and the dense layout
 * (no wasted bytes). Layouts (hexagon-sim one-hot maps, bit-exact vs a reference, see sim/gemm_u8_sim.c):
 *   A(s, k): byte 32*s + k of a 2 KB crouton (s < 64, k < 32) -- plain row-major [64][32];
 *   W(k, c): byte 128*(k/4) + 4*c + k%4 of a 1 KB block (same as the non-cm int8 weight);
 *            weight :deep = two such blocks back to back (columns 0..31, then 32..63) -> both accumulators;
 *   C(s, c): byte 32*s + c of a 2 KB tile, = min(255, floor(max(acc, 0) * s_c / 512)) with s_c the fp16 low
 *            half of table word c (exact for power-of-two s_c; other scales can come out 1 LSB low).
 * K accumulates over consecutive instructions (one per 32-channel crouton) until the store. */
static inline void hmx_blk_mac_u8cm(const uint8_t* a, const uint8_t* w, int ktiles) {
  for (int k = 0; k < ktiles; k++)
    __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3) }" ::"r"(a + (size_t)k * 2048), "r"(0x7ff),
                     "r"(w + (size_t)k * 1024), "r"(0x3ff)
                     : "memory");
}
/* 64 output columns: w = ktiles x 2 KB (per K block: columns 0..31, then 32..63); store twice (hmx_blk_store_u8cm) */
static inline void hmx_blk_mac_u8cm_deep(const uint8_t* a, const uint8_t* w, int ktiles) {
  for (int k = 0; k < ktiles; k++)
    __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3):deep }" ::"r"(a + (size_t)k * 2048),
                     "r"(0x7ff), "r"(w + (size_t)k * 2048), "r"(0x7ff)
                     : "memory");
}
static inline void hmx_blk_store_u8cm(void* c) {
  __asm__ volatile("mxmem(%0,%1):after:cm:sat.ub = acc" ::"r"(c), "r"(0) : "memory");
}

static inline void hmx_blk_store_f16(void* c) { __asm__ volatile("mxmem(%0,%1):after.hf = acc" ::"r"(c), "r"(0) : "memory"); }
static inline void hmx_blk_store_u16(void* c) {
  __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1" ::"r"(c), "r"(0) : "memory");
}
static inline void hmx_blk_store_u8(void* c) { __asm__ volatile("mxmem(%0,%1):after:sat.ub = acc" ::"r"(c), "r"(0) : "memory"); }
#endif /* __hexagon__ */
#endif
