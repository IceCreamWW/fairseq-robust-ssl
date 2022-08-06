# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# see /mnt/lustre/sjtu/home/czy97/sid/voxceleb/Speaker-Recognition_master/utils/dataset_h5.py:149 for musan mixing

import itertools
import logging
import os
import sys
import re
import pdb
from typing import Any, List, Optional, Union

import numpy as np
import kaldiio
import soundfile as sf
import random
from scipy import signal

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)

def norm_wav(wav):
    #  norm wav value to [-1.0, 1.0]
    norm = max(np.absolute(wav))
    if norm > 1e-5:
        wav = wav / norm
    return wav

def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class HubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        noise_manifest_paths: Optional[List[str]] = None, # ["/path/to/musan/noise", "/path/to/musan/speech"] (uttid, path, num_samples)
        rir_manifest_paths: Optional[List[str]] = None, # ["/path/to/musan/noise", "/path/to/musan/speech"]
        noise_snr: Optional[List[float]] = [5, 10], # [lower, upper]
        num_noise: List[int] = [1, 2], #[lower, upper]
        aug_types: List[str] = None, # ["reverberate", "additive"]
        train: bool = True,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )

        self.train = train
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

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_names[index])

        if re.match(r".*\.ark:\d+", wav_path): # kaldi ark style audio path
            cur_sample_rate, wav = kaldiio.load_mat(wav_path)
        else:
            wav, cur_sample_rate = sf.read(wav_path)

        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        item = {"id": index, "source": wav, "source_aug": None, "label_list": labels}

        # if self.train and self.aug_types is not None and len(self.aug_types) > 0:
        if self.aug_types is not None and len(self.aug_types) > 0:
            aug_type = random.choice(self.aug_types)
            wav_aug = wav.numpy()
            if aug_type == "additive":
                wav_aug = self.additive_noise(wav_aug)
            elif aug_type == "reverberate":
                wav_aug = self.reverberate(wav_aug)
            else:
                raise RuntimeError(f"unknown aug type {aug_type}")
            wav_aug = torch.from_numpy(wav_aug).float()
            wav_aug = self.postprocess(wav_aug, 16000)
            item["source_aug"] = wav_aug
        return item

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        audios_aug = [s["source_aug"] for s in samples]
        if audios_aug[0] is not None:
            collated_audios_aug, *_ = self.collater_audio(audios_aug, audio_size)
        else:
            collated_audios_aug = None

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "source_aug": collated_audios_aug, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def read_audio(self, path):
        if re.match(r".*\.ark:\d+", path): # kaldi ark style audio path
            sample_rate, wav = kaldiio.load_mat(path)
        else:
            wav, sample_rate = sf.read(path)
        return wav, sample_rate

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


