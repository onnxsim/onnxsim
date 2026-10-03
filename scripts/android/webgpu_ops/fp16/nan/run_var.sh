# usage: [INPUT=file] [WARM= ITERS=] run_var.sh MODEL TAGS...   (on phone)
D=/data/local/tmp/wg-nan
cd $D && export LD_LIBRARY_PATH=.
name=$1; shift
[ -n "$INPUT" ] && export INPUT_BIN=$D/$INPUT
rm -rf dump/${name}_cpu; mkdir -p dump/${name}_cpu
./bench_in m/$name.onnx cpu 1 1 threads=4 dump=dump/${name}_cpu >/dev/null 2>&1
for tag in "$@"; do
  rm -rf dump/${name}_$tag; mkdir -p dump/${name}_$tag
  echo -n "$name $tag: "; ./bench_in m/${name}_$tag.onnx webgpu ${WARM:-15} ${ITERS:-8} dump=dump/${name}_$tag 2>&1 | grep -E "RESULT|ERR|rror" | tail -1 | sed 's/.*median=/median=/; s/ mean.*//'
done
