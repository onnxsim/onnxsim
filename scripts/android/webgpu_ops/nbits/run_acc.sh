D=/data/local/tmp/wg-nbits
cd $D; export LD_LIBRARY_PATH=.
rm -rf dump; mkdir dump
for model in resnet50 yolo11n yolo26n; do
  sh=""; [ $model = resnet50 ] && sh="shape=pixel_values:1,3,224,224"
  mkdir -p dump/${model}_cpu; ./bench m/$model.onnx cpu 1 1 threads=4 dump=dump/${model}_cpu $sh >/dev/null 2>&1
  mkdir -p dump/${model}_base; ./bench m/$model.onnx webgpu 1 1 dump=dump/${model}_base $sh enableInt64=1 >/dev/null 2>&1
  for v in q8_32 q8_128 q4_32 q4_128; do
    mkdir -p dump/${model}_$v; ./bench m/${model}_$v.onnx webgpu 1 1 dump=dump/${model}_$v $sh enableInt64=1 >/dev/null 2>&1
  done
  # profile + optimized graph for the base and two quantized variants
  for v in base q8_32 q4_32; do
    f=m/$model.onnx; [ $v != base ] && f=m/${model}_$v.onnx
    rm -rf prof_${model}_$v; mkdir prof_${model}_$v
    PROFILE=prof_${model}_$v/p ./bench $f webgpu 1 2 $sh enableInt64=1 save_opt=/data/local/tmp/wg-nbits/opt_${model}_$v.onnx >/dev/null 2>&1
  done
done
