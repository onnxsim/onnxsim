# on the phone: chainrun2.sh  -> "<tag> <opts> N2 <ms> N10 <ms>" for every chain model, with the current opt-ins
cd /data/local/tmp/wg-attr2; export LD_LIBRARY_PATH=.
for f in m/chain/chain_*_N2.onnx; do
  t=$(basename $f _N2.onnx | sed 's/^chain_//')
  for cfg in "ORT_WEBGPU_CONV_TEXDIRECT=1" "ORT_WEBGPU_CONV_TEXDIRECT=0"; do
    a=$(env $cfg ./bench m/chain/chain_${t}_N2.onnx webgpu 40 40 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2)
    b=$(env $cfg ./bench m/chain/chain_${t}_N10.onnx webgpu 40 40 enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2)
    echo "$t $cfg N2 $a N10 $b"
  done
done
