# usage: run_acc.sh MODEL INPUT_FILE TAGS...  -> for each tag: default acc and ACC32
D=/data/local/tmp/wg-nan
cd $D && export LD_LIBRARY_PATH=.
name=$1; inp=$2; shift 2
export INPUT_BIN=$D/$inp
rm -rf dump/${name}_cpu; mkdir -p dump/${name}_cpu
./bench_in m/$name.onnx cpu 1 1 threads=4 dump=dump/${name}_cpu >/dev/null 2>&1
for tag in "$@"; do for acc in 0 1; do
  if [ $acc = 1 ]; then export ORT_WEBGPU_F16_ACC32=1; else unset ORT_WEBGPU_F16_ACC32; fi
  rm -rf dump/${name}_${tag}_a$acc; mkdir -p dump/${name}_${tag}_a$acc
  echo -n "$name $tag acc32=$acc: "; ./bench_in m/${name}_$tag.onnx webgpu ${WARM:-12} ${ITERS:-8} dump=dump/${name}_${tag}_a$acc 2>&1 | grep -E "RESULT|rror" | tail -1 | sed 's/.*median=/median=/; s/ mean.*//'
done; done
