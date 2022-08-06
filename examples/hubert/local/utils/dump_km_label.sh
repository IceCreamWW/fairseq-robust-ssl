#!/usr/bin/env bash
## bash local/utils/dump_km_label.sh --feat_dir dump/raw/ls_960/feat_from_hubert_base_ls960_L9/ --split valid --km_path download_models/hubert_base_ls960_L9_km500.bin --nj 4 --dst dump/raw/ls_960/km_from_hubert_base_ls960_L9/ --device 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feat_dir=dump/raw/ls_960/feat_from_hubert_base_ls960_L9/
dst=dump/raw/ls_960/km_from_hubert_base_ls960_L9/
split=valid
km_path=download_models/hubert_base_ls960_L9_km500.bin
cmd=run.pl
layer=9
nj=4
device=

. ./path.sh
. utils/parse_options.sh
max_job_num=$((nj-1))


mkdir -p $dst
SECONDS=0
[ -f $dst/km_model.pt ] && rm $dst/km_model.pt
ln -s `realpath $km_path` $dst/km_model.pt
if [ -z $device ]; then
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=2 python simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} $nj JOB $dst
else
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} $nj JOB $dst
fi

echo "dump feature done, ${SECONDS}s Elapsed. "

