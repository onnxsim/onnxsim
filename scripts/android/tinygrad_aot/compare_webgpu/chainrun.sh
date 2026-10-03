D=/data/local/tmp/wg-bench
cd $D && export LD_LIBRARY_PATH=.
for f in m/chain/*.onnx; do for t in 0 16; do echo -n "$(basename $f .onnx) tex=$t "; ORT_WEBGPU_WINO_TEX=$t ./bench $f webgpu 80 40 2>&1 | grep RESULT | tail -1 | grep -o 'median=[0-9.]*ms'; done; done
