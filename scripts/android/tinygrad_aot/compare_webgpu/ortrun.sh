D=/data/local/tmp/wg-bench
cd $D && export LD_LIBRARY_PATH=.
for r in 1 2; do
for spec in "resnet50 resnet50 shape=pixel_values:1,3,224,224" "yolo11n yolo11n" "resnet50_f16 resnet50 shape=pixel_values:1,3,224,224" "yolo11n_f16 yolo11n"; do
  set -- $spec; f=$1; shift 2
  case $f in *_f16) env="" ;; *) env="" ;; esac
  for t in 0 16; do
    case $f in *_f16) [ $t = 16 ] && continue;; esac
    echo -n "ORT $f tex=$t: "; ORT_WEBGPU_WINO_TEX=$t ./bench m/$f.onnx webgpu 60 25 $@ 2>&1 | grep RESULT | tail -1 | grep -o 'median=[0-9.]*ms'
  done
done
done
