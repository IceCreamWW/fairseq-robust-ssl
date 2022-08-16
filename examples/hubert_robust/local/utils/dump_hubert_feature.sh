#!/usr/bin/env bash
## local/utils/dump_hubert_feature.sh --src data/ls_960/ --split valid --model simple_kmeans/hubert_base_ls960.pt --layer 9 --nj 4 --dst dump/raw/ls_960/feat_from_hubert_base_ls960

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src=data/ls_960
dst=dump/raw/ls_960
split=valid
cmd=run.pl
model=download_models/hubert_base_ls960.pt
layer=9
nj=4
device=


. ./path.sh
. utils/parse_options.sh
max_job_num=$((nj-1))

SECONDS=0
[ -f $dst/model.pt ] && rm $dst/model.pt
ln -s `realpath $model` $dst/model.pt
if [ -z $device ]; then
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=2 python simple_kmeans/dump_hubert_feature.py $src $split $model $layer $nj JOB $dst
else
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python simple_kmeans/dump_hubert_feature.py $src $split $model $layer $nj JOB $dst
fi

echo "dump feature done, ${SECONDS}s Elapsed. "

