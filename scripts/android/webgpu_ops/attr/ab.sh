# on the phone: ab.sh BASE.onnx VAR.onnx WARM ITERS ROUNDS   (ABAB interleaved medians)
cd /data/local/tmp/wg-attr; export LD_LIBRARY_PATH=.
base=$1; var=$2; warm=$3; iters=$4; rounds=$5
r=0
while [ $r -lt $rounds ]; do
  echo -n "A "; ./bench m/$base webgpu $warm $iters enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*'
  echo -n "B "; ./bench m/$var webgpu $warm $iters enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*'
  r=$((r+1))
done
