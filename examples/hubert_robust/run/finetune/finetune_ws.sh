#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=base_10h_ws_norm
# config_dir=config/finetune/4gpu/50k/
config_dir=config/finetune/4gpu
expdir=exp/finetune_10h
hubert_tag=
data=data/ll_10
init=download_models/hubert_base_ls960.pt

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1

. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
init=`realpath $init`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_finetune_$(basename "${data}")_from_official_hubert_base_ls960_ws
fi

hubert_exp=$expdir/${hubert_tag}
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 fairseq-hydra-train \
# CUDA_VISIBLE_DEVICES=3 fairseq-hydra-train \
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${data} \
  model.w2v_path=${init} hydra.run.dir=${hubert_exp}

