#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=hubert_base_librispeech
expdir=exp
hubert_tag=
data=data/ls_960
label=dump/raw/ls_960/km_from_mfcc_ls960/

. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_train_$(basename "${data}")_label_$(basename "${label}")
fi
hubert_exp=$expdir/${hubert_tag}

CUDA_VISIBLE_DEVICES=3 fairseq-hydra-train \
  --config-dir config/pretrain/debug \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' model.label_rate=100 hydra.run.dir=${hubert_exp}
