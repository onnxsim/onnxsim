D=/data/local/tmp/wg-nan
cd $D && export LD_LIBRARY_PATH=.
for m in "$@"; do
  for acc in 0 1; do
    export INPUT_BIN=$D/iso/${m}_in.bin
    if [ $acc = 1 ]; then export ORT_WEBGPU_F16_ACC32=1; else unset ORT_WEBGPU_F16_ACC32; fi
    rm -rf diso/${m}_$acc; mkdir -p diso/${m}_$acc
    ./bench_in iso/$m.onnx webgpu 2 3 dump=diso/${m}_$acc >/dev/null 2>&1
  done
done
