cd /data/local/tmp/wg-attr2; export LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT=1
for r in 1 2 3; do
for m in yolo11n yolo26n; do
  for v in "$m" "${m}_go" "${m}_go_cc"; do echo -n "$v "; ./bench m/$v.onnx webgpu 3 20 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'; done
done
echo -n "r50 plain "; ./bench m/resnet50.onnx webgpu 3 20 shape=pixel_values:1,3,224,224 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'
echo -n "r50 capture "; ./bench m/resnet50.onnx webgpu 3 20 shape=pixel_values:1,3,224,224 enableInt64=1 enableGraphCapture=1 iobind=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms\|ERR.*'
for m in yolo11n yolo26n; do echo -n "$m capture "; ./bench m/$m.onnx webgpu 3 20 enableInt64=1 enableGraphCapture=1 iobind=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms\|ERR.*'; echo -n "$m plain+iobind "; ./bench m/$m.onnx webgpu 3 20 enableInt64=1 iobind=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'; done
done
