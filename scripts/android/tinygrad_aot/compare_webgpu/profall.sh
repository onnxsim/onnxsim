R=/data/local/tmp/codex-demo-app-tinygrad
cd $R; export LD_LIBRARY_PATH=$R/py/lib TG_PROFILE_ALL=1
for b in r50_fp16:r50_in.bin r50_fp32img:r50_in.bin y11_fp16:y11_in.bin y11_fp32img:y11_in.bin; do
  d=${b%%:*}; i=${b##*:}
  ./tg_cl_bench_all cmp/$d cmp/$i cmp/$d/q 60 1 > cmp/$d/profile_all.txt 2>&1
done
# ORT profile for resnet50 (current default build)
D=/data/local/tmp/wg-bench; cd $D; export LD_LIBRARY_PATH=.; unset TG_PROFILE_ALL
rm -rf prcmp && mkdir prcmp
PROFILE=prcmp/r50 ./bench m/resnet50.onnx webgpu 40 3 shape=pixel_values:1,3,224,224 >/dev/null 2>&1
PROFILE=prcmp/y11 ./bench m/yolo11n.onnx webgpu 40 3 >/dev/null 2>&1
