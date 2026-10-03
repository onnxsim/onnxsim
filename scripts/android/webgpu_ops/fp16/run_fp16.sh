D=/data/local/tmp/wg-fp16
cd $D && export LD_LIBRARY_PATH=. && rm -rf dump && mkdir dump
runm() { name=$1; warm=$2; iters=$3; shift 3
  echo "== $name warm=$warm iters=$iters"
  mkdir -p dump/${name}_cpu && ./bench m/$name.onnx cpu 1 1 threads=4 dump=dump/${name}_cpu "$@" >/dev/null 2>&1
  for v in _f32 _f16 _w16; do
    mkdir -p dump/$name$v
    f=m/$name.onnx; [ $v != _f32 ] && f=m/$name$v.onnx
    echo -n "$name$v: "; ./bench $f webgpu $warm $iters dump=dump/$name$v "$@" 2>&1 | grep RESULT | tail -1
  done; }
runm resnet50 100 20 shape=pixel_values:1,3,224,224
runm yolo11n 80 20
runm sam_l0_enc 15 8
runm rtdetr_pre 18 10
