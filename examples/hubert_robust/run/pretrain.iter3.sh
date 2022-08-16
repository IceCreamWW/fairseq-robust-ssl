#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=hubert_base_librispeech
config_dir=config/pretrain/iter3/8gpu
# config_dir=config/pretrain/iter2/1gpu
expdir=exp
hubert_tag=
data=data/ls_960
# label=dump/raw/ls_960/km_from_hubert_base_iter1_L6_ls960/
label=dump/raw/ls_960/official_iter2/km_from_hubert_base_ls960_L7

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_train_$(basename "${data}")_label_$(basename "${label}")_iter3
fi
hubert_exp=$expdir/${hubert_tag}

# CUDA_VISIBLE_DEVICES=2 fairseq-hydra-train \
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' model.label_rate=50 hydra.run.dir=${hubert_exp}

