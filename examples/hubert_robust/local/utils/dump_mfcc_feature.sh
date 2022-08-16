#!/usr/bin/env bash
## local/utils/dump_mfcc_feature.sh --src data/ls_960/ --split valid --nj 4 --dst dump/raw/ls_960/mfcc

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src=data/ls_960
dst=dump/raw/ls_960/mfcc
split=valid
cmd=run.pl
nj=4

. path.sh
. utils/parse_options.sh
max_job_num=$((nj-1))


SECONDS=0
$cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
    python simple_kmeans/dump_mfcc_feature.py $src $split $nj JOB $dst

echo "dump feature done, ${SECONDS}s Elapsed. "

