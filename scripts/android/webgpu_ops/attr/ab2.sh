# on the phone: ab2.sh BASE.onnx VAR.onnx WARM ITERS ROUNDS [extra bench args]   (ABAB interleaved medians, current opt-ins)
cd /data/local/tmp/wg-attr2; export LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT=1
base=$1; var=$2; warm=$3; iters=$4; rounds=$5; shift 5
r=0
while [ $r -lt $rounds ]; do
  echo -n "A "; ./bench m/$base webgpu $warm $iters "$@" enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*'
  echo -n "B "; ./bench m/$var webgpu $warm $iters "$@" enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*'
  r=$((r+1))
done
