# usage: run_bis.sh MODEL_FILE INPUT_FILE OUTDIR
D=/data/local/tmp/wg-nan
cd $D && export LD_LIBRARY_PATH=.
export INPUT_BIN=$D/$2
rm -rf $3 && mkdir -p $3
./bench_in m/$1 webgpu 1 1 dump=$3 2>&1 | tail -1
ls $3 | wc -l
