tinygrad Adreno OpenCL vs ORT WebGPU, one phone session (see ../../WEBGPU_SURVEY.md, section "tinygrad Adreno OpenCL vs the ORT WebGPU stack").

- `export.sh MODEL OUT IN.bin IMAGE FLOAT16` -- on-phone export (`aot/export_cl.py --adreno`), run through `phone-run`.
- `runtg.sh`, `profall.sh` -- `tg_cl_bench` timing (150 iterations) and a per-kernel GPU profile (`TG_PROFILE_ALL=1` prints every call in order).
- `ortrun.sh`, `chainrun.sh` -- ORT WebGPU `bench` runs (whole models; chains of 2 and 10 identical convs for per-conv wall time).
- `results/` -- the per-kernel profiles and the chain timings.
Input: resnet50 with a static 1x3x224x224 input (random normal*0.5 float), yolo11n float twin (random 0..255 float NHWC).
