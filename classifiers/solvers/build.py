import logging
from typing import Optional

import torch
from torch import nn

from classifiers.solvers.schedulers import MultiStepLRScheduler

log = logging.getLogger(__name__)

from classifiers.solvers.schedulers import (
    make_cosine_anneal,
    reduce_lr_on_plateau,
    warmup_cosine_decay,
    create_multistep_lr_scheduler,
)

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,
    "multistep_lr": create_multistep_lr_scheduler,  # reduce lr by a factor of gamma at each milestone,
    "cosine_annealing_warm_restarts": make_cosine_anneal,  # quickly decays lr then spikes back up for a "warm restart" https://paperswithcode.com/method/cosine-annealing
    "warmup_cosine_decay": warmup_cosine_decay,  # linearly increases lr then decays it with a cosine function
    "reduce_lr_on_plateau": reduce_lr_on_plateau,
}


def get_optimizer_params(
    model: torch.nn.Module, strategy: str = "swin", backbone_lr: Optional[float] = None
):
    """Extract the traininable parameters from the model in different groupes; allows us to spceicfy
    different learning rates for different groups of parameters

    Strategies:
        swin: TODO
        simmim:
        separate_backbone: separate the backbone parameters from the rest of the model and use
                           a different learning rate for the backbone
    """

    # NOTE: swin and simmmim swin appear to have the same method for setting up the parameter groups
    if strategy == "swin":
        # extract the parameters which should not have weight decay applied
        log.info(">>>>>>>>>> Build Optimizer for Pre-training Stage")
        skip = {}
        skip_keywords = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
            log.info(f"No weight decay: {skip}")
        if hasattr(model, "no_weight_decay_keywords"):
            skip_keywords = model.no_weight_decay_keywords()
            log.info(f"No weight decay keywords: {skip_keywords}")

        param_dicts = get_simmim_pretrain_param_groups()
    else:
        raise ValueError(f"Unknown parameter group strategy: {strategy}")

    return param_dicts


def get_simmim_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    """Separate the trainable parameters into two groups: ones with weight decay and ones
    without weight decay.

    Specifically
    We do not apply weight decay to the following parameters:
        1. all 1D parameters (e.g., bias, LayerNorm weight, LayerNorm bias, BatchNorm weight)
        2. encoder.mask_token, encoder.absolute_pos_embed, encoder.relative_position_bias_table

    We do apply weight decay to:
        1. Conv/Linear weights

    """
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)

    logger.info(f"No decay params: {no_decay_name}")
    logger.info(f"Has decay params: {has_decay_name}")
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


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

    log.info(optimizer)

    # TODO: impelement configs for warmup_cosine_decay; i think this is already done?
    # Build scheduler
    
    #### start here, try to build scheduler
    if scheduler_name in scheduler_map:
        scheduler_params["num_epochs"] = num_epochs
        scheduler_params["steps_per_epoch"] = num_steps_per_epoch
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
