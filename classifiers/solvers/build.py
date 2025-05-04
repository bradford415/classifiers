from collections.abc import Iterable

import torch

from classifiers.solvers.schedulers import (make_cosine_anneal,
                                            reduce_lr_on_plateau,
                                            warmup_cosine_decay)

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


def build_solvers(
    model_params: Iterable,
    optimizer_params: dict[str, any],
    scheduler_params: dict[str, any],
):
    """Builds the optimizer and learning rate scheduler based on the provided parameters
    from solver.config

    Args:
        optimizer_params: the parameters used to build the optimizer
        scheduler_params: the parameters used to build the learning rate scheduler
        optimizer: the optimizer used during training
    """
    optimizer_name = optimizer_params["name"]
    scheduler_name = scheduler_params["name"]

    # TODO: use a parameters field instead of this; Delete name key so we can easily unpack the parameters
    del optimizer_params["name"]
    del scheduler_params["name"]

    # Build optimizer
    if optimizer_name in optimizer_map:
        optimizer = optimizer_map[optimizer_name](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # TODO: impelement configs for warmup_cosine_decay
    # Build scheduler
    if scheduler_name in scheduler_map:
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
