from flashlight.lib.text.decoder import KenLM
from flashlight.lib.text.dictionary import create_word_dict, load_words

# path_to_lexicon="/mnt/lustre/sjtu/home/zkz01/tools/fairseq/examples/wav2vec/decoding/babel_cantonese_lexicon.lst"
path_to_lexicon="/mnt/lustre/sjtu/home/zkz01/tools/fairseq/examples/wav2vec/decoding/librispeech_lexicon.lst"
# path_to_lm="/mnt/lustre/sjtu/home/zkz01/tools/fairseq/examples/wav2vec/decoding/babel_cantonese_4gram.bin"
path_to_lm="/mnt/lustre/sjtu/home/ww089/espnets/espnet-text-adapt/egs2/librispeech/asr1/data/en_token_list/bpe_unigram10000/ls960_4gram.arpa"

lexicon = load_words(path_to_lexicon)
word_dict = create_word_dict(lexicon)
lm = KenLM(path_to_lm, word_dict)
