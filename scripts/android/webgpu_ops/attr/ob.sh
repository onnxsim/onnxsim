# on the phone: ob.sh  -> "<name> K2 <median> K32 <median>" for every ob_*_K2.onnx in m/
cd /data/local/tmp/wg-attr; export LD_LIBRARY_PATH=.
for f in m/ob_*_K2.onnx; do
  b=$(basename $f _K2.onnx)
  a=$(OPT_LEVEL0=1 ./bench_attr m/${b}_K2.onnx webgpu 40 60 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*')
  c=$(OPT_LEVEL0=1 ./bench_attr m/${b}_K32.onnx webgpu 40 60 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*')
  echo "$b K2 $a K32 $c"
done
