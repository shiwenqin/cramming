"""Instantiate backend objects in a congruent format.

The interface is made to be compliant with the deepspeed interface.

"""
import torch

from .torch_default import initialize_torch

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


def load_backend(model, dataset, tokenizer, cfg_train, cfg_impl, elapsed_time=0.0, setup=_default_setup):
    if cfg_impl.name == "torch-default":
        return initialize_torch(model, dataset, tokenizer, cfg_train, cfg_impl, elapsed_time, setup=setup)
    else:
        raise ValueError(f"Invalid backend {cfg_impl.name} given.")
