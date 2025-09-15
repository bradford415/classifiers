from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from classifiers.solvers.schedulers import (
    make_cosine_anneal,
    reduce_lr_on_plateau,
    warmup_cosine_decay,
)

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,
    "cosine_annealing_warm_restarts": make_cosine_anneal,  # quickly decays lr then spikes back up for a "warm restart" https://paperswithcode.com/method/cosine-annealing
    "warmup_cosine_decay": warmup_cosine_decay,  # linearly increases lr then decays it with a cosine function
    "reduce_lr_on_plateau": reduce_lr_on_plateau,
}


def get_optimizer_params(
    model: torch.nn.Module, strategy: str = "all", backbone_lr: Optional[float] = None
):
    """Extract the traininable parameters from the model in different groupes; allows us to spceicfy
    different learning rates for different groups of parameters

    Strategies:
        all: extract all traininable parameters from the model and use the same learning rate
        separate_backbone: separate the backbone parameters from the rest of the model and use
                           a different learning rate for the backbone
    """

    parameters = [
        param for name, param in model.named_parameters() if param.requires_grad
    ]
    param_dicts = [{"params": parameters}]

    return param_dicts


def build_solvers(
    model: nn.Module,
    num_epochs: int,
    num_steps_per_epoch: int,
    optimizer_config: dict[str, any],
    scheduler_config: dict[str, any],
):
    """Builds the optimizer and learning rate scheduler based on the provided parameters
    from solver.config

    Args:
        optimizer_params: the parameters used to build the optimizer
        scheduler_params: the parameters used to build the learning rate scheduler
        optimizer: the optimizer used during training
    """
    optimizer_name = optimizer_config["name"]
    scheduler_name = scheduler_config["name"]

    optimizer_params = optimizer_config["params"]
    scheduler_params = scheduler_config["params"]

    model_params = get_optimizer_params(model)

    # Build optimizer
    if optimizer_name in optimizer_map:
        optimizer = optimizer_map[optimizer_name](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # TODO: impelement configs for warmup_cosine_decay
    # Build scheduler
    if scheduler_name in scheduler_map:
        scheduler_params["num_epochs"] = num_epochs
        scheduler_params["steps_per_epoch"] = num_steps_per_epoch
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
