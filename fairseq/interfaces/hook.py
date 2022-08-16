import sys
from typing import Callable, List, Dict, Tuple, Union

import logging
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TOLERABLE_SEQLEN_DIFF = 5


class Hook:
    def __init__(self, module_path, transform, unique_identifier=None):
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)


class initHook(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for hook in instance.hooks:
            if hook.handler is None:
                instance._register_hook_handler(hook)
        return instance


class Hookable(nn.Module, metaclass=initHook):
    def __init__(
        self,
        hooks: List[Tuple] = None,
        hook_postprocess: Callable[
            [List[Tuple[str, Tensor]]], List[Tuple[str, Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Args:
            hooks: each Tuple is an argument list for the Hook initializer
        """
        super().__init__()
        self.hooks: List[Hook] = [Hook(*hook) for hook in hooks] if hooks else []
        self.hook_postprocess = hook_postprocess
        self._hook_hiddens: List[Tuple(str, Tensor)] = []

    def remove_all_hooks(self):
        for hook in self.hooks:
            hook.handler.remove()
        self.hooks.clear()

    def remove_hook(self, unique_identifier: str):
        updated_hooks = []
        for hook in self.hooks:
            if hook.unique_identifier == unique_identifier:
                hook.handler.remove()
            else:
                updated_hooks.append(hook)
        self.hooks = updated_hooks

    def add_hook(self, *args, **kwargs):
        hook = Hook(*args, **kwargs)
        self._register_hook_handler(hook)
        self.hooks.append(hook)

    def _register_hook_handler(self, hook: Hook):
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):
            logger.error(
                f"[Hookable] - {hook.module_path} is not a valid nn.Module. Skip.",
                file=sys.stderr,
            )
            return

        if callable(hook.handler):
            logger.error(
                f"[Hookable] - Existing hook handler for {hook.unique_identifier} is found. Remove the existing one.",
                file=sys.stderr,
            )
            hook.handler.remove()

        def generate_hook_handler(hiddens: List, hook: Hook):
            def hook_handler(self, input, output):
                hiddens.append((hook.unique_identifier, hook.transform(input, output)))

            return hook_handler

        hook.handler = module.register_forward_hook(
            generate_hook_handler(self._hook_hiddens, hook)
        )

#     def __call__(self, wavs: List[Tensor], *args, **kwargs):
# 
#         pdb.set_trace()
#         self._hook_hiddens.clear()
# 
#         result = super().__call__(wavs, *args, **kwargs) or {}
#         assert isinstance(result, dict)
# 
#         if len(self._hook_hiddens) > 0:
#             if (
#                 result.get("_hidden_states_info") is not None
#                 or result.get("hidden_states") is not None
#                 or result.get("last_hidden_state") is not None
#             ):
#                 logger.error(
#                     "[Hookable] - If there are registered hooks, '_hidden_states_info', 'hidden_states', and "
#                     "'last_hidden_state' are reserved and should not be included in child class's return dict.",
#                     file=sys.stderr,
#                 )
#                 raise ValueError
# 
#             hook_hiddens = self._hook_hiddens.copy()
#             self._hook_hiddens.clear()
# 
#             if callable(self.hook_postprocess):
#                 hook_hiddens = self.hook_postprocess(hook_hiddens)
# 
#             result["_hidden_states_info"], result["hidden_states"] = zip(*hook_hiddens)
#             result["last_hidden_state"] = result["hidden_states"][-1]
# 
#             for layer_id, hidden_state in enumerate(result["hidden_states"]):
#                 result[f"hidden_state_{layer_id}"] = hidden_state
# 
#         return result

