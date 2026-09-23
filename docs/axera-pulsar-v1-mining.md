# Mining the previous-generation (AX620A / Pulsar v1) toolchain for mcode semantics

The AX650 `teng2` compute program has resisted decoding by build-diffing, and the one
AX650-side disassembler (`ax650npu_cmodel.so`) is Sentinel-LDK license-gated
(`docs/axera-teng2-toolchain-mining.md`). This checks whether the previous chip generation
(AX620A, used by Sipeed's MaixIII AX-Pi board) ships less-protected tooling from the same
ISA family. It does, and more than expected.

Everything below comes from publicly redistributed artifacts:

- `sipeed/axpi_bsp_sdk` (GitHub): Sipeed's Axera-authorized AX620A BSP. Earlier work
  (`scripts/axera/README.md`, "Public prior-generation API headers") used only its headers.
- `sipeed/pulsar:0.6.1.20` (Docker Hub, 1.25 GB compressed, 3.15 GB unpacked): the Pulsar v1
  toolchain image. Axera's own docs (`AXERA-TECH/pulsar-docs`, quick start) distribute the
  same toolchain as `axera_neuwizard_v0.6.1.x.tar.gz` via Baidu Pan; this is Sipeed's public
  mirror of it.

Nothing license-gated or obfuscated was bypassed. What is protected is listed as such.

## Verdict: AX620A and AX650 mcode share one record format and register space

**AX650's verb records are register writes.** An AX620A compiled model (`resnet50.joint`
from the BSP) is a tar containing `part_0.neu`, a FlatBuffers `dot-neu` (namespace
`manhattan::dot_neu`, see below). Its mcode is a 230,752-byte vector made only of 8-byte
records:

```
[verb][unit][reg lo][reg hi][value32]
a1 01 40 05  01000000
a1 01 80 02  10f70100      write reg 0x0280 = 0x0001f710
```

All 28,844 records are one of four kinds: `a1` (16,021), `a2` (4,285) each followed by a
`00`-prefixed payload record (4,286), and `a3` (4,252). Every `a1` targets one of 69
16-byte-aligned register addresses (`0x0080`, `0x0280`, `0x0290`, `0x02a0`, `0x02b0` ...).

AX650's `mcode.py` already parses verb records as `[verb][xx][field][bank][32-bit operand]`,
with `field` a multiple of `0x10`. The AX620A layout shows what those bytes are: `field` and
`bank` are the low and high bytes of a **16-bit register address**, and the record is a
32-bit register write. AX650 verb records all have byte 1 = `0x00`; AX620A's all have
`0x01`, which is the only format difference found. Example from an AX650 Relu build:
`a1 00 80 02 00010000` = write `0x100` to register `0x0280`. The register spaces overlap:
of 53 registers an AX650 Relu writes with `a1` and 69 in the AX620A ResNet50, 13 are shared,
including the dominant DMA triplet `0x0280`/`0x0290`/`0x02a0` and `0x0150`, `0x0160`,
`0x0180`, `0x0190`, `0x02b0`-`0x02d0`, `0x0380`-`0x03a0`.

The difference is that AX650 streams interleave the compressed short units this project
decoded separately (`S`/`B`/tag forms). AX620A's stream is plain 8-byte records, so AX650's
tokenizer covers only 0.1% of it as verbs. AX620A mcode is effectively the uncompressed
form of the same register-write program.

**A first value match.** The same `.neu` embeds the compiler's own IR (below). Its first
command is a load whose `to_ocm.addr = 2060544` and `to_ocm.subsize = 64`; the mcode's first
`0x0280` write is `0x1f710 = 2060544 >> 4` and the first `0x0290` write is `4 = 64 >> 4`.
Addresses and sizes are stored in 16-byte units. A naive in-order correlation over the whole
program matched poorly (at most 21% for any IR field / register pair): 28,844 records drive
27,253 IR commands, far too few for a register burst per command. The `a2` + payload pairs
most likely fetch descriptor blocks stored elsewhere in the `.neu` (vectors of 160-180 KB
sit beside the mcode). That mapping is the natural next step and was not done here.

## Finding 1: an ungated on-device mcode generator, with its instruction IR schema

`msp/out/lib/libax_npu_cv_kit.so` (AX620A BSP, ARM32, 2.4 MB, dynamic symbols only) builds
mcode at runtime for CV operators (resize, CSC, CCM, affine, matmul, l2-normalize, alpha
blend). `npu_cv::CmdsGenerator` assembles `npu_instructor::Cmd` protobuf messages and calls
`assembler_generate_mcode`. The assembler is statically linked in: no external assembler
dependency, no license strings. Operator names confirm TENG runs elementwise work:
`MatrixMultiplyRunner::do_multiply_on_teng`, `L2NormalizeRunner::do_l2_normalize_on_teng`,
`AlphaBlendWithMaskRunner::do_blend_on_teng`.

Protobuf-generated code embeds its schema, and both files extract cleanly as
`FileDescriptorProto`s (`cmd.proto`, `pulsar_model.proto`, package `npu_instructor`,
proto3):

- `Cmd` is `id`, `eu_idx`, and a oneof: `sync`, `ld`, `st`, `mv`, `warp`, `remap`, `blend`,
  `ccm`, `fetch`, `conv`, `itp`, `ydrc`.
- `LdCmd`/`StCmd`/`MvCmd`: DMA between DDR and OCM (`Detail` = addr, step, subsize, times,
  addrsize), with a `DMAStream` of sub-IPs applied in flight: padding, resize (area and
  bilinear), YUV format, CCM, affine, bayer, pack/unpack, extract, byte-crop, and `fma`.
- `FMASubIP`: `input_stream`s, a `crossbar_list`, `compare` constants, `Stats` (SUM, MAX,
  ARGMAX, MIN, ARGMIN, EQUAL_VALUE), and a DRC set. This is the TENG FMA datapath.
- `ConvCmd`: inputs, `add_input`, output OCM, strides, pooling, padding, `w_bit`,
  `output_bit`, LUT (EXP, SWISH, TANH), `group_conv_num`, `fp32_add_mode`.
- `SyncCmd`: `sync_eus` (sync id, EU index).
- `PulsarModel`: `cmds`, `base_addrs`, `params_bins`, `runtime_vars`, virtual-NPU mode, and
  `librosetta_params_bin`.

**The compiled `.neu` carries this IR.** Field 8 of `part_0.slim_neu`'s root table is zlib
data that decompresses to a 5.9 MB serialized `PulsarModel` with 27,253 commands: `sync`
10,423, `ld` 6,713, `conv` 3,399, `fetch` 3,353, `mv` 3,349, `st` 16, on EUs 3 and 0. Each
carries operand values and op names (`op_21226`, `{"no_crc": false, "from_ddr": {...}}`).
Every AX620A model built by Pulsar v1 therefore ships ground-truth IR next to its assembled
mcode: a paired corpus, not a black box.

## Finding 2: an unstripped simulator with full DWARF, including TENG register structures

The Pulsar v1 image's compiler Python (`eruptor`, `magma_tools`, `neuwizard`,
`pulsar_compiler`, `super_pulsar`, `pfuns`) is **Pyarmor-obfuscated** (every module starts
with `__pyarmor__(...)`). Not pursued.

`/root/python_modules/npu_simulator/` is not obfuscated. It holds x86-64 executables
`run_neu.x86.ax620a` (a host build of the device-side dot-neu runner) and
`npu_simulator.ax620a`, and a Python extension `npu_simulator_ax620a` that imports without
any license check (exports `compile`, `version`, `FinalCheckError`). Strings contain no
Sentinel/HASP/license markers.

`pulsar_compiler/cfuns/npu_sim_par_ax620a.cpython-36m-x86_64-linux-gnu.so` (43 MB) is **not
stripped** (11,876 symbols) and carries **full DWARF** (15.8 MB `.debug_info`). It contains:

- The assembler (`assembler_export.cpp`, `npu_generate_mcode.cpp`, `dot_neu_pack.cpp`) and the
  IR-to-hardware translators: `load_fetch(Cmd, potato_cfg_t&)`,
  `load_affine(SubIP, teng_stream_cfg_t&, ...)`, `load_resize`, `load_unpack`, `load_input`,
  `load_remap`, `load_blend`.
- A C functional model of the hardware: `teng_top`, `fma_top`, `fma_ld`, `fma_st`,
  `teng_transpose_sw_impl`, `teng_haar_sw_impl`, `pack_sw_impl`, `teng_fbc_sw_impl`, and
  config checkers (`check_fma_cfg`, `potato_cfg_check`).
- Complete struct layouts, down to bitfields, via `pahole`. `teng_job_cfg_t` (5,568 bytes:
  `eu_id`, `dma_cmd_type`, wait/update job configs, a skip condition) wraps
  `teng_stream_cfg_t` (5,488 bytes, 37 members): DMA control, rdma/wdma common and
  channel-0 configs, enables, mux selects, pack/extract/bayer/resize/affine/unpack/padding/
  bilinear/YUV/CCM/statistic/haar/transpose/reorder blocks, `fma_cfg_t` (3,208 bytes: split,
  crossbar, data-gen, result and stats outputs), three FMA read-DMA configs, write-DMA 1,
  jump configs, FBC, CRC. Leaves are bitfields, for example:

  ```c
  typedef struct {                 /* fma_adder_cfg_t, 48 bytes */
      uint8_t  en:1; uint8_t op:2; uint8_t a_sel:1; uint8_t b_sel:1;
      uint32_t comp_greater, comp_equal, comp_less;
      uint32_t consts[8];
  } fma_adder_cfg_t;
  typedef struct {                 /* teng_stats_cfg_t, 24 bytes */
      uint32_t cycle_max; uint8_t fun_type:3; uint8_t abs_en:1;
      uint32_t HW_max, C_type, K_fp, cmp_value;
  } teng_stats_cfg_t;
  ```
  Also named: `potato_cfg_t` (the fetch/conv-side engine, matching AX650's
  `npu_potato_set_queue`), `warp_config_t`, `remap_config_t`, `npu_wb_config_t`,
  `fma_mult_cfg_t`, `fma_div_cfg_t`, `fma_data_gen_cfg_t`.
- 210 FlatBuffers vtable enumerators (`VT_*`) and generated builders for the container
  formats: `manhattan::dot_neu::CreateModelFileInfo`, `CreateRuntimePatchInfoList`, and
  `rosetta::runtime_var::*` (affine/DRC/mask/WB-clip tables). These give field names for
  the `.neu` FlatBuffers, the ancestor of the AX650 mcode tail this project reverse-engineered
  as `mcode.tail_tables`.

No register-address constant table was found by name. Addresses are most likely produced
by the packing code (`npu_generate_mcode.cpp`), which was not disassembled here.

## What this does and does not give the AX650 effort

It gives, for the first time and without any license issue:

1. The meaning of the verb record: a 32-bit write to a 16-bit register address. That turns
   the AX650 `teng2` problem from "unknown ISA" into "which configuration field lives at
   which register". It is consistent with the AX650 driver's `cmdq_write_instruct` symbol.
2. A register-level vocabulary for TENG: every field the hardware exposes, with bit widths.
3. A paired IR/mcode corpus: any Pulsar v1 build embeds the `Cmd` protobuf beside its mcode.
4. A runnable, ungated x86 simulator and assembler for the same ISA family (AX620A).

It does not directly give AX650's register map: the chips share 13 of the registers seen so
far, and AX650 has more EU types (MAU, SDMA) and a compressed stream encoding. Two concrete
next steps:

- **Map fields to registers on AX620A**, where nothing is compressed or gated: build tiny
  models with Pulsar v1, vary one IR operand, and diff the mcode (or trace the `a2` descriptor
  fetches). `load_*` and the DWARF structs say which fields exist; the diff says where each
  lands.
- **Carry the shared registers over to AX650**: `0x0280`/`0x0290`/`0x02a0` are the OCM
  destination address, size and step of a DMA in 16-byte units on AX620A. Check whether the
  same holds in AX650 Relu/Add builds, where these registers are already known to carry
  `C`-linear values (`docs/axera-dma-queue.md`).

## Reproduction

Scratch artifacts are in `/home/takecheeze/npu-scratch/t_pulsar_v1` (not committed):
`fetch_bsp.py` (downloads the BSP files), `extract_proto.py` (pulls the embedded
`FileDescriptorProto`s from `libax_npu_cv_kit.so` and renders them), `fb_walk.py`
(schema-less FlatBuffers walk of a `.neu`), `parse_neu.py` / `correlate.py` (parse the
embedded `PulsarModel`, correlate IR fields with register writes), and `probe*.sh` (run in
`sipeed/pulsar:0.6.1.20` with `docker run --rm -v DIR:/w --entrypoint /bin/bash IMAGE
/w/probe.sh`). Struct layouts: `pahole -C teng_stream_cfg_t
npu_sim_par_ax620a.cpython-36m-x86_64-linux-gnu.so`.
