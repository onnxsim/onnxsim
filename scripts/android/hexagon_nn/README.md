# Inference with the installed Hexagon NN runtime

Run a small FP32 dense layer directly on the Android phone's compute DSP using
its existing `libhexagon_nn_stub.so` and `libhexagon_nn_skel.so`. This provides a
hardware inference check on devices whose firmware rejects custom SDK-signed
DSP libraries. It needs an Android NDK and ADB, without a Hexagon SDK or SNPE SDK.

```bash
python scripts/android/hexagon_nn/run.py \
  --serial YOUR_ADB_SERIAL \
  --ndk /path/to/android-sdk/ndk/27.2.12479018
```

The runner targets Android arm64 API 28 and builds on a Linux x86_64 host. The
phone must expose the vendor NN libraries and allow FastRPC access from the ADB
shell. Root was available on the tested Xperia; this does not establish support
for unprivileged apps or other firmware images. Select another stub location,
DSP search path, or FastRPC URI with `--stub`, `--dsp-library-path`, or `--uri`.
Host builds use a temporary directory; device files are staged in a unique
`/data/local/tmp` directory and removed afterward. Vendor files are not changed.
Missing libraries, unsupported graph operations, execution errors, and numerical
mismatches fail the command. No host inference fallback is implemented.

The graph computes `y = ReLU(xW + b)` using `INPUT → MatMul_f → Add_f → Relu_f →
OUTPUT`, with four FP32 inputs and three outputs. Four input vectors exercise
positive values, negative values, and zero. Every output's shape and value is
checked against a CPU reference (absolute tolerance `1e-6`). Operation IDs are
queried from the device rather than assumed from an enum. The minimal ABI
declarations follow the public [nnlib interface](https://github.com/XiaoMi/nnlib/blob/df7ad3c1b235ecfeafa6ac9448be462e3f51ec6a/interface/hexagon_nn.idl).
Only the host executable is built; the DSP kernels come from the installed library.

This is a native graph inference check, not an ONNX importer, a benchmark, or
validation of all NN operators. It does not claim quantized HVX acceleration.

## Observed device result

Sony Xperia XZ2 Compact, SDM845/Hexagon v65, Android 15: installed runtime version
`0x20e02`; graph preparation and all four executions succeeded. All 12 output
values matched exactly:

| Input | Output |
| --- | --- |
| `[1, 2, 3, 4]` | `[8.75, 0.5, 6.5]` |
| `[-1, 0, 2, -3]` | `[0, 0.75, 0]` |
| `[0, 0, 0, 0]` | `[0.25, 0, 1]` |
| `[3, -2, 0.5, 1]` | `[7.5, 0, 0]` |

## Why the installed DSP library is trusted

On this Xperia, the NN skeleton contains a production certificate chain ending
at Qualcomm **QDSR Root CA**, whose SHA-256 certificate fingerprint is
`b66f913fdca0bc2173b4fe73c1db04fc80adc2ca3c9ad0e91cd017bc2983892a`.
DSP FARF diagnostics report the active image root beginning `b6,6f,91,3f`,
matching that certificate. SDK-generated signatures instead use the Qualcomm
OpenDSP test chain, with root fingerprint
`0c09760d72b2e99689f004bf8ef0ebbb83cf4511bdf6ccf8389b33a35e42ab09`.
The loader reports `signature does not match image root`, `Testsig Enabled: No`,
and `Testsig file valid: No` for our custom library. Local SDK signature
validation does not imply acceptance by the device's firmware.

Trust comes from firmware authentication, not the filename, the `/vendor` path,
or ADB root access. These signing-policy observations are specific to this
firmware; other devices can have different roots and development policies.
