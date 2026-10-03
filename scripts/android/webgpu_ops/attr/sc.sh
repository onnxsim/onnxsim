# on the phone: chains of stand-ins; prints medians (ms) for N=2 and N=22
cd /data/local/tmp/wg-attr; export LD_LIBRARY_PATH=.
for sn in s256x32 s128x256 s64x512; do for kind in mul slice_cat; do for r in 1 2; do
  a=$(./bench m/sc_${sn}_${kind}_2.onnx webgpu 60 40 2>&1 | tail -1 | grep -o 'median=[0-9.]*')
  b=$(./bench m/sc_${sn}_${kind}_22.onnx webgpu 60 40 2>&1 | tail -1 | grep -o 'median=[0-9.]*')
  echo "$sn $kind N2 $a N22 $b"
done; done; done
