import os
import sys
import re
from typing import List, Optional
from tqdm import tqdm
import argparse

import numpy as np
import kaldiio
import soundfile as sf
import random
from scipy import signal

import torch
import torch.nn.functional as F
from fairseq.data.audio.hubert_dataset import norm_wav

import pdb

# python local/utils/make_noisy_dataset.py --manifest_path data/ls_test_clean/test.manifest --noise_manifest_paths /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/music.manifest /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/noise.manifest --num_noise 1 1 --noise_snr 5 10 --aug_types additive --output_dir /mnt/lustre02/scratch/sjtu/home/ww089/rawdata/librispeech/test_clean_music_noise_5_10db

class AugmentDataset:
    def __init__(
        self,
        manifest_path: str,
        noise_manifest_paths: List[str] = None, # ["/path/to/musan/noise", "/path/to/musan/speech"] (uttid, path, num_samples)
        rir_manifest_paths: List[str] = None, # ["/path/to/musan/noise", "/path/to/musan/speech"]
        noise_snr: List[float] = [5, 10], # [lower, upper]
        num_noise: List[int] = [1, 1], #[lower, upper]
        aug_types: List[str] = None, # ["reverberate", "additive"]
        output_dir: str = None
    ):
        self.uttid_and_audios = []
        with open(manifest_path) as fp:
            for line in fp:
                uttid, audio, *_ = line.strip().split()
                self.uttid_and_audios.append((uttid, audio))

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.aug_types = aug_types
        self.noise_type2id_and_length_and_path = {}
        if noise_manifest_paths is not None:
            for idx, noise_manifest_path in enumerate(noise_manifest_paths):
                noises = []
                with open(noise_manifest_path) as fp:
                    for line in fp:
                        uttid, path, num_samples = line.strip().split()
                        noises.append((uttid, int(num_samples), path))
                self.noise_type2id_and_length_and_path[idx] = noises
        self.noise_snr = noise_snr
        self.num_noise = num_noise

        self.rir_type2id_and_length_and_path = {}

        if rir_manifest_paths is not None:
            for idx, rir_manifest_path in enumerate(rir_manifest_paths):
                rirs = []
                with open(rir_manifest_path) as fp:
                    for line in fp:
                        uttid, path, num_samples = line.strip().split()
                        rirs.append((uttid, int(num_samples), path))
                self.rir_type2id_and_length_and_path[idx] = rirs

        pass

    def __call__(self):
        for uttid, audio_path in tqdm(self.uttid_and_audios):
            wav = self.get_audio(audio_path).numpy()
            kaldiio.save_ark(
                os.path.join(self.output_dir, "data_wav.ark"),
                {uttid: (wav, 16000)},
                scp=os.path.join(self.output_dir, "wav.scp"),
                append=True,
                write_function=f"soundfile_wav",
            )


    def get_audio(self, audio_path):
        wav, _ = self.read_audio(audio_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav)

        aug_type = random.choice(self.aug_types)
        wav = wav.numpy()
        if aug_type == "additive":
            wav = self.additive_noise(wav)
        elif aug_type == "reverberate":
            wav = self.reverberate(wav)
        else:
            raise RuntimeError(f"unknown aug type {aug_type}")
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav)
        return wav

    def postprocess(self, wav):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        return wav

    def get_random_chunk_start(self, data_len, chunk_len):
        adjust_chunk_len = min(data_len, chunk_len)
        chunk_start = random.randint(0, data_len - adjust_chunk_len)
        return chunk_start, adjust_chunk_len

    def additive_noise(self, audio, audio_sr=16000):
        '''
        :param audio: numpy array, (audio_len,)
        '''
        audio = audio.astype(np.float32)
        audio = norm_wav(audio)
        audio_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        audio_len = audio.shape[0]
        if audio_sr == 8000:
            audio_len = audio_len * 2

        noise_type = random.randint(0, len(self.noise_type2id_and_length_and_path) - 1)
        noise_idx_list = random.sample(self.noise_type2id_and_length_and_path[noise_type], random.randint(self.num_noise[0], self.num_noise[1]))

        noise_list = []
        for noise_id, noise_len, path in noise_idx_list:
            chunk_start, chunk_len = self.get_random_chunk_start(noise_len, audio_len)
            noise, _ = self.read_audio(path)
            noise = np.resize(noise, (audio_len,)).astype(np.float32)
            noise = norm_wav(noise)
            if audio_sr == 8000:
                noise = noise[::2]

            noise_snr = random.uniform(self.noise_snr[0], self.noise_snr[1])
            noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
            noise_list.append(np.sqrt(10 ** ((audio_db - noise_db - noise_snr) / 10)) * noise)

        return np.sum(np.stack(noise_list), axis=0) + audio

    def reverberate(self, audio, audio_sr):
        '''
        :param audio: numpy array, (audio_len,)
        '''

        audio = audio.astype(np.float32)
        audio = norm_wav(audio)
        audio_len = audio.shape[0]

        rir_type = random.randint(0, len(self.rir_type2id_and_length_and_path) - 1)
        rir_id, rir_len, path = random.choice(self.rir_type2id_and_length_and_path[rir_type])
        rir_audio = self.read_audio(path)

        if audio_sr == 8000:
            rir_audio = rir_audio[::2]
        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))

        return signal.convolve(audio, rir_audio, mode='full')[:audio_len]

    def read_audio(self, path):
        if re.match(r".*\.ark:\d+", path): # kaldi ark style audio path
            sample_rate, wav = kaldiio.load_mat(path)
        else:
            wav, sample_rate = sf.read(path)
        return wav, sample_rate

if __name__ == "__main__":

    # python local/utils/make_noisy_dataset.py
    parser = argparse.ArgumentParser(description='mix noise with data')
    parser.add_argument('-i', '--manifest_path', type=str,
                        help='manifest path [uttid audio ...]')
    parser.add_argument('-n', '--noise_manifest_paths', type=str, nargs='*',
                        help='noise manifest paths')
    parser.add_argument('-r', '--rir_manifest_paths', type=str, nargs='*', default=None,
                        help='rir manifest paths')
    parser.add_argument('-s', '--noise_snr', type=int, nargs='+',
                        help='noise snr [lower, upper]')
    parser.add_argument('-c', '--num_noise', type=int, nargs='+',
                        help='num noise [lower, upper]')
    parser.add_argument('-a', '--aug_types', type=str, nargs='+', choices=["additive", "reverberate"],
                        help='[additive, reverberate]')
    parser.add_argument('-o', '--output_dir', help='dump dir for data_wav.ark and wav.scp')

    args = parser.parse_args()

    augment_dataset = AugmentDataset(args.manifest_path, args.noise_manifest_paths, args.rir_manifest_paths, args.noise_snr, args.num_noise, args.aug_types, args.output_dir)
    augment_dataset()


