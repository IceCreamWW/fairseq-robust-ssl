module load cuda/10.1 gcc/6.4.0 cmake/3.12.0 sox/14.4.2

# activate conda env
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /mnt/lustre/sjtu/home/ww089/espnets/espnet-enh1/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate fairseq-robust-ssl

export PATH=$PWD/utils/:$PATH
