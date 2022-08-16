python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/decode \
  --config-name infer_viterbi \
  task.data=data/ls_dev_clean/ \
  task.normalize=false \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
  decoding.decoder.lexicon=/path/to/lexicon \
  decoding.decoder.lmpath=/path/to/arpa
