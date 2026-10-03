# usage: export.sh <model> <outdir> <input.bin> <IMAGE> <FLOAT16>
R=/data/local/tmp/codex-demo-app-tinygrad
cd $R
. $R/phone_env.sh
export PARALLEL=0 TG_CLC=$R/clc DEV=CL JIT_BATCH_SIZE=0 IMAGE=$4 FLOAT16=$5
rm -rf $R/cmp/$2
time $R/py/python3 aot/export_cl.py cmp/$1 cmp/$2 --adreno --check cmp/$3 2>&1 | tail -15
ls cmp/$2 | head
