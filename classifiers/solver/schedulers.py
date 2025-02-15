import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.

    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        """Initialize the warmup cosine decay scheduler

        Args:
            optimizer: torch optimizer to update w/ the learning rate to be updated
            warmup_steps: Linearly increase learning rate from 0 to 1 over this many steps; after
                          warmup_steps, learning rate decreases from 1 to 0 following a cosine curve
            t_total: total number of steps to decay to 0 over; cosine decay will start after warmup_steps
            cycles:
            last_epoch:
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """Lambda function passed into LambdaLR to update the learning rate; this function return is multiplied
        by the INITIAL learning rate to get the new learning rate
        """
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


def warmup_cosine_decay(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
):
    """Builds the warmup cosine decay lr scheduler

    Args:
        optimizer: torch optimizer to update w/ the learning rate to be updated
        warmup_steps: Linearly increase learning rate from 0 to 1 over this many steps; after
                        warmup_steps, learning rate decreases from 1 to 0 following a cosine curve
        total_steps: total number of steps to decay to 0 over; cosine decay will start after warmup_steps
    """
    return WarmupCosineSchedule(
        optimizer, warmup_steps, t_total=total_steps, cycles=0.5, last_epoch=-1
    )


def make_cosine_anneal(optimizer: torch.optim.Optimizer):
    """Builds the cosine annealing lr scheduler"""

    # TODO: hardcode parameters once internet comes back
    # Number of steps to decay over between two warm restarts; when current_step = max_steps
    # the lowest lr value (eta_min) is used then
    # TODO: figure out what this value should be
    max_steps = 50

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=max_steps
    )


def reduce_lr_on_plateau(
    optimizer: torch.optim.Optimizer,
    mode: str = "min",
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
):
    """Builds the reduce lr on plateau scheduler

    Args:
        mode: "min" reduces the lr when a value does not decrease any more;
              "max" reduces the lr when a value does not increase any more
        factor: multiplicative factor to reduce the learning rate by; this updates the
                current learning rate as opposed to something like LambdaLR which updates
                the initial learning rate
        patience: number of epochs to wait for no imporovement before reducing the learning rate
        threshold: threshold to determine if the value has improved
                   i.e., the value must improve by more than this threshold to count as an improvement

    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
    )
