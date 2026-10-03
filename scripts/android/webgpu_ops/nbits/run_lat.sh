D=/data/local/tmp/wg-nbits
cd $D; export LD_LIBRARY_PATH=.
rm -f lat.txt
for r in 1 2 3; do
for model in resnet50 yolo11n yolo26n; do
  sh=""; [ $model = resnet50 ] && sh="shape=pixel_values:1,3,224,224"
  for v in base q8_32 q8_64 q8_128 q4_32 q4_64 q4_128; do
    f=m/$model.onnx; [ $v != base ] && f=m/${model}_$v.onnx
    med=$(./bench $f webgpu 60 15 $sh enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2)
    echo "$model $v $r $med" >> lat.txt
  done
done; done
cat lat.txt
