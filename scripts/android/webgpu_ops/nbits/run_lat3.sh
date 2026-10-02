D=/data/local/tmp/wg-nbits
cd $D; export LD_LIBRARY_PATH=.
rm -f lat3.txt
for r in 1 2; do
for model in resnet50 yolo11n yolo26n; do
  sh=""; [ $model = resnet50 ] && sh="shape=pixel_values:1,3,224,224"
  for v in base q8_32fb q8_32a4 q8_64a4 q8_128a4 q4_32a4 q4_64a4 q4_128a4; do
    f=m/$model.onnx; [ $v != base ] && f=m/${model}_$v.onnx
    med=$(./bench $f webgpu 60 15 $sh enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2)
    echo "$model $v $r $med" >> lat3.txt
  done
done; done
cat lat3.txt
