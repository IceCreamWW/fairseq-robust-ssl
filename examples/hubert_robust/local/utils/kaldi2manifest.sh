#!/usr/bin/env bash
## local/utils/kaldi2manifest.sh --src /mnt/lustre/sjtu/shared/data/asr/rawdata/LibriSpeech/dump/raw/train_960 --dst data/ls_960 --split train --audio_path_prefix /mnt/lustre/sjtu/shared/data/asr/rawdata/LibriSpeech/dump/raw/org/train_960_sp/data/

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src=
dst=
audio_path_prefix=
split=
stage=1
stop_stage=2

. path.sh
. utils/parse_options.sh
mkdir -p $dst/kaldi


SECONDS=0
src_tmp=$dst/kaldi/$split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: copy and fix kaldi style dir"
    [ -d ${src_tmp} ] && rm -r ${src_tmp}
    cp -r $src ${src_tmp}
    utils/fix_data_dir.sh ${src_tmp}
    # [ -f ${src_tmp}/utt2dur ] || utils/data/get_utt2dur.sh ${src_tmp}
    [ -f ${src_tmp}/utt2dur ] && [ "$(wc -l < ${src_tmp}/utt2dur)" -eq "$(wc -l < ${src_tmp}/wav.scp)"  ] || python local/utils/get_ark_duration.py --wav_scp ${src_tmp}/wav.scp  > ${src_tmp}/utt2dur
fi

src=${src_tmp}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: prepare tsv file"
    awk -v audio_path_prefix=${audio_path_prefix} '{sub(audio_path_prefix,"",$1); print $1}' $src/wav.scp > $dst/$split.tsv

    if [ -z ${audio_path_prefix} ]; then
        cut -d" " -f2 $src/wav.scp > $dst/.${split}.tsv.tmp
    else
        sed "s:$audio_path_prefix::" $src/wav.scp > $dst/.${split}.tsv.tmp
    fi
    paste $dst/.$split.tsv.tmp <(cut -d" " -f2 $src/utt2dur) > $dst/${split}.tsv && rm $dst/.${split}.tsv.tmp
    sed -i "1s:^:${audio_path_prefix}\n:" $dst/$split.tsv
fi

echo "Manifest Preparation Done, ${SECONDS}s Elapsed. "
