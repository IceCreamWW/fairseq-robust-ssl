set -o pipefail
set -e

model=exp/hubert_base_librispeech_train_ls_960_label_km100_from_mfcc/final.pt
layer=2
nj=4
device=0
# dumpdir=dump/raw/ls_960/reproduce/iter1/
dumpdir=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert_robust/dump/raw/ls_960/reproduce/iter1
# kmeansdir=models/kmeans/reproduce/iter1/
kmeansdir=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert_robust/models/kmeans/reproduce/iter1

mkdir -p $dumpdir
mkdir -p $kmeansdir

. ./path.sh
. utils/parse_options.sh

max_job_num=$((nj-1))

mkdir -p $dumpdir/feat_from_hubert_base_ls960_L$layer
mkdir -p $kmeansdir

# echo "dumping train feature for layer $layer"
# local/utils/dump_hubert_feature.sh --src data/ls_960/ --split train --model $model --layer $layer --nj $nj --dst $dumpdir/feat_from_hubert_base_ls960_L$layer --device $device
# 
# echo "dumping valid feature for layer $layer"
# local/utils/dump_hubert_feature.sh --src data/ls_960/ --split valid --model $model --layer $layer --nj $nj --dst $dumpdir/feat_from_hubert_base_ls960_L$layer --device $device
# 
# echo "learning kmeans for layer $layer"
# OPENBLAS_NUM_THREADS=20 python simple_kmeans/learn_kmeans.py $dumpdir/feat_from_hubert_base_ls960_L$layer train $nj $kmeansdir/ls_960_hubert_base_L${layer}_500.km 500 --percent 0.1

echo "dump train km label for layer $layer"
local/utils/dump_km_label.sh --feat_dir $dumpdir/feat_from_hubert_base_ls960_L$layer/ --split train --km_path $kmeansdir/ls_960_hubert_base_L${layer}_500.km --nj $nj --dst $dumpdir/km_from_hubert_base_ls960_L$layer/ --device $device

for i in `seq 0 $max_job_num`; do cat $dumpdir/km_from_hubert_base_ls960_L$layer/train_${i}_${nj}.km; done > $dumpdir/km_from_hubert_base_ls960_L$layer/train.km

echo "dump valid km label for layer $layer"
local/utils/dump_km_label.sh --feat_dir $dumpdir/feat_from_hubert_base_ls960_L$layer/ --split valid --km_path $kmeansdir/ls_960_hubert_base_L${layer}_500.km --nj $nj --dst $dumpdir/km_from_hubert_base_ls960_L$layer/ --device $device

for i in `seq 0 $max_job_num`; do cat $dumpdir/km_from_hubert_base_ls960_L$layer/valid_${i}_${nj}.km; done > $dumpdir/km_from_hubert_base_ls960_L$layer/valid.km

rm $dumpdir/feat_from_hubert_base_ls960_L$layer/*npy

# python measure_teacher_quality.py data/ls_960/ $dumpdir/km_from_hubert_base_ls960_L$layer/ km --phn_dir dump/raw/ls_960/phone_frame_align/ --phn_sets dev_clean dev_other --upsample 2 --verbose

