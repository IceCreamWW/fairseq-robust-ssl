#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=hubert_base_librispeech_with_init
# config_dir=config/pretrain/robust/1gpu #debug
config_dir=config/pretrain/robust/8gpu
expdir=exp
hubert_tag=
data=data/ls_960
label=dump/raw/ls_960/official_iter2/km_from_hubert_base_ls960_L7/

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_train_noisy_audio_clean_target_$(basename "${data}")_label_$(basename "${label}")
fi
hubert_exp=$expdir/${hubert_tag}

# CUDA_VISIBLE_DEVICES=2 fairseq-hydra-train \
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' \
  task.noises='["/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/music.manifest", "/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/noise.manifest"]' \
  task.num_noise='[1,2]' task.aug_types='["additive"]' task.noise_snr='[5,10]' \
  model.label_rate=50 hydra.run.dir=${hubert_exp} \
  model.init=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/download_models/hubert_base_ls960.pt

