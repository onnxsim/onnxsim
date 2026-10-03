cd /data/local/tmp/wg-attr2; export LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT=1
for r in 1 2 3; do
for spec in "resnet50 shape=pixel_values:1,3,224,224" "yolo11n" "yolo26n"; do
  set -- $spec; m=$1; sh=$2
  echo -n "$m sync plain   "; ./bench m/$m.onnx webgpu 3 20 $sh enableInt64=1 iobind=1 sync=tiny.onnx 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'
  echo -n "$m sync capture "; ./bench m/$m.onnx webgpu 3 20 $sh enableInt64=1 enableGraphCapture=1 iobind=1 sync=tiny.onnx 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'
done; done
