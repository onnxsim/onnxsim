D=/data/local/tmp/wg-cc-tune
cd $D; export LD_LIBRARY_PATH=.
./bench m/yolo11n.onnx webgpu 60 5 enableInt64=1 >/dev/null 2>&1
for r in 1 2 3; do for m in yolo11n yolo26n; do
 for cfg in "0 1" "1 1" "0 0" "1 0"; do set -- $cfg
  echo -n "$m fused=$1 texdirect=$2 "; ORT_WEBGPU_CONCAT_CONV=$1 ORT_WEBGPU_CONV_TEXDIRECT=$2 ./bench m/$m.onnx webgpu 25 14 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'
 done
done; done
