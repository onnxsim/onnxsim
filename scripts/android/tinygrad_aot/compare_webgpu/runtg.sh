R=/data/local/tmp/codex-demo-app-tinygrad
cd $R
export LD_LIBRARY_PATH=$R/py/lib
for b in r50_fp16:r50_in.bin r50_fp32img:r50_in.bin r50_fp32buf:r50_in.bin y11_fp16:y11_in.bin y11_fp32img:y11_in.bin y11_fp32buf:y11_in.bin; do
  d=${b%%:*}; i=${b##*:}
  echo "== $d"
  ./tg_cl_bench cmp/$d cmp/$i cmp/$d/o 150 0 2>&1 | grep -v "^$" | tail -3
  ./tg_cl_bench cmp/$d cmp/$i cmp/$d/p 10 1 > cmp/$d/profile.txt 2>&1
  head -3 cmp/$d/profile.txt
done
