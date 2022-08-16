# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask

from fairseq.interfaces.hook import Hookable

import pdb


@dataclass
class HubertAsrWsConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to hubert model"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    normalize_ws: bool = field(
        default=False,
        metadata={"help": "whether to normalize before weighted sum"},
    )
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class HubertCtcWsConfig(HubertAsrWsConfig):
    pass


@register_model("hubert_ctc_ws", dataclass=HubertCtcWsConfig)
class HubertCtcWs(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcWsConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcWsConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoderWs(cfg, task)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class HubertEncoderWs(FairseqEncoder, Hookable):
    def __init__(self, cfg: HubertAsrWsConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )


        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        FairseqEncoder.__init__(self, pretrain_task.source_dictionary)
        Hookable.__init__(self)

        d = w2v_args.model.encoder_embed_dim
        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if task.target_dictionary is not None:
            self.proj = Linear(d, len(task.target_dictionary))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

        self.layer_num = len(self.w2v_model.encoder.layers) + 1
        self.normalize = cfg.normalize_ws
        self.weights = nn.Parameter(torch.zeros(self.layer_num))

        if len(self.hooks) == 0:
            module_name = "self.w2v_model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.w2v_model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))
            self.hook_postprocess = postprocess


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, source_aug=None, **kwargs):

        if source_aug is not None:
            source = source_aug

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        assert not ft, "ws is not designed to finetune hubert encoder"

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

        feature_info, feature = zip(*self._hook_hiddens)
        x = self._weighted_sum(feature)
        self._hook_hiddens.clear()

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature)
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],))

        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
