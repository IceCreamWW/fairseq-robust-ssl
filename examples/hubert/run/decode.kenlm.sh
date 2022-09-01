#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail
expdir=exp/finetune_10h/base_10h_finetune_ll_10_from_official_hubert_base_ls960_layer_10
checkpoint=checkpoint_best.pt
test_sets="dev_clean"
tag='kenlm'
device=0
lexicon=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert_robust/dump/raw/ls_960/models/lm/lexicon.lst
lmpath=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert_robust/dump/raw/ls_960/models/lm/ngram.bin

. ./path.sh
. utils/parse_options.sh

expdir=`realpath $expdir`
checkpoint=`realpath $expdir/checkpoints/$checkpoint`

for test_set in $test_sets; do
    CUDA_VISIBLE_DEVICES=$device python ../speech_recognition/new/infer.py \
      --config-dir config/decode \
      --config-name infer_kenlm \
      task.data=`realpath data/ls_$test_set` \
      task.normalize=false \
      common_eval.results_path=$expdir/decode_$tag/$test_set \
      common_eval.path=$checkpoint \
      dataset.gen_subset=test \
      decoding.lexicon=$lexicon \
      decoding.lmpath=$lmpath
done

