D=/data/local/tmp/wg-cc-tune
cd $D; export LD_LIBRARY_PATH=.
T11="128:128:20=2,2,32,4;256:256:20=2,2,32,4;384:256:20=2,2,32,4;512:256:20=2,2,32,4;192:128:40=2,2,16,4;64:64:40=2,2,16,4;48:64:160=2,2,128,1"
T26="128:128:20=2,2,8,16;256:256:20=2,2,8,16;384:256:20=2,2,8,16;512:256:20=2,2,8,16;48:64:160=2,2,32,4"
./bench m/yolo11n.onnx webgpu 60 5 enableInt64=1 >/dev/null 2>&1
run() { # model label fused td table
  echo -n "$1 $2 "; ORT_WEBGPU_CONCAT_CONV=$3 ORT_WEBGPU_CONV_TEXDIRECT=$4 ORT_WEBGPU_CONCATCONV_TABLE="$5" ./bench m/$1.onnx webgpu 25 14 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*ms'
}
for r in 1 2 3 4 5; do
 run yolo11n unfused_td1 0 1 ""
 run yolo11n fused_default_td1 1 1 ""
 run yolo11n fused_table_td1 1 1 "$T11"
 run yolo11n fused_default_td0 1 0 ""
 run yolo11n fused_table_td0 1 0 "$T11"
 run yolo26n unfused_td1 0 1 ""
 run yolo26n fused_default_td1 1 1 ""
 run yolo26n fused_table_td1 1 1 "$T26"
 run yolo26n fused_default_td0 1 0 ""
 run yolo26n fused_table_td0 1 0 "$T26"
done
for m in yolo11n yolo26n; do
  rm -rf dc_$m dg_$m; mkdir dc_$m dg_$m
  ./bench m/$m.onnx cpu 1 1 threads=4 dump=dc_$m >/dev/null 2>&1
  if [ $m = yolo11n ]; then T="$T11"; else T="$T26"; fi
  ORT_WEBGPU_CONCAT_CONV=1 ORT_WEBGPU_CONV_TEXDIRECT=1 ORT_WEBGPU_CONCATCONV_TABLE="$T" ./bench m/$m.onnx webgpu 1 1 dump=dg_$m enableInt64=1 >/dev/null 2>&1
  mkdir -p acc_$m/dg_unf; rm -rf dgu_$m; mkdir dgu_$m
  ORT_WEBGPU_CONCAT_CONV=0 ORT_WEBGPU_CONV_TEXDIRECT=1 ./bench m/$m.onnx webgpu 1 1 dump=dgu_$m enableInt64=1 >/dev/null 2>&1
done
