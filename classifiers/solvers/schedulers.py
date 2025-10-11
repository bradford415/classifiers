import math

import torch
from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class MultiStepLRScheduler(Scheduler):
    """Reduces learning rate a specfici milestones by a given factor gammm
    e.g., milestones = [20, 40] epochs; this is different then StepLR because this reduces
    the learning rate at regularly intervals e.g., every 20 epochs
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones,
        gamma=0.1,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs=True,
    ) -> None:
        """Initalize the learning rate scheduler

        Args:
            optimizer: torch optimizer to update w/ the learning rate to be updated
            milestones: list of epochs/steps to reduce the learning rate at
            gamma: multiplicative factor to reduce the learning rate by at each milestone
            warmup_t: number of epochs/steps to linearly increase the learning rate over before decaying it
            warmup_lr_init: initial learning rate to start the linear warmup from; if 0 then starts from 0
            t_in_epochs: if True, milestones and warmup_t are in epochs; if False, they are in steps
        """
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        # self.base_values is defined as the base learning rate for each param group;
        # e.g., in the default case self.basE_values = [0.0001, 0.0001] for two param groups
        # (one with weight decay and one without)

        # calcluate the number of warmup steps
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [
                v * (self.gamma ** bisect_right(self.milestones, t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.

    Works in two phases:
        1. Linearly increases learning rate from 0 to the initial lr over `warmup_steps`
        training steps.
        2. Decreases learning rate from inital lr to 0 over remaining `t_total - warmup_steps` steps following a cosine curve.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_min_lr: float = 1e-6,
        num_cycles: float = 0.5,
        last_epoch=-1,
    ):
        """Initialize the warmup cosine decay scheduler

        Args:
            optimizer: torch optimizer to update w/ the learning rate to be updated
            warmup_steps: linearly increase learning rate from 0 to 1 over this many steps; after
                          warmup_steps, learning rate decreases from 1 to 0 following a cosine curve
            total_steps: total number of training steps/epochs; cosine decay will start after warmup_steps
                         and will slowly decay to 0
            warmup_min_lr: the minimum learning rate to use during warmup; this is needed because at the first
                           epoch the learning rate would be 0 so we need to add a small value to it
                           or else the weights wouldn't update for the first epoch
                           TODO: might need to modify this for the main lr too after warmup
            num_cycles: the number of waves in the cosine schedule. Defaults to 0.5; practically, I think this controls how
                        steep the drop in learning rate is (i.e., the higher number the steeper the drop)
            (decrease from the max value to 0 following a half-cosine).
            last_epoch: index of the last epoch when resuming training; defaults to -1
        """
        self.warmup_steps = warmup_steps
        self.t_total = total_steps
        self.warmup_min_lr = warmup_min_lr
        self.num_cycles = num_cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """Lambda function passed into LambdaLR to update the learning rate; this function return is multiplied
        by the INITIAL learning rate to get the new learning rate

        Args:
            step: the current training step or epoch
        """
        if step < self.warmup_steps:
            return max(
                self.warmup_min_lr, float(step) / float(max(1.0, self.warmup_steps))
            )
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )


def warmup_cosine_decay(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    num_epochs: int,
    steps_per_epoch: int,
    num_cycles: float = 0.5,
    warmup_min_lr: float = 1e-6,
):
    """Builds the warmup cosine decay lr scheduler

    Args:
        optimizer: torch optimizer to update w/ the learning rate to be updated
        warmup_steps: Linearly increase learning rate from 0 to 1 over this many steps; after
                      warmup_steps, learning rate decreases from 1 to 0 following a cosine curve
        total_steps: total number of steps to decay to 0 over; cosine decay will start after warmup_steps
        warmup_min_lr: the minimum learning rate to use during warmup; this is needed because at the first
                       epoch the learning rate would be 0 so we need to add a small value to it
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    ### start here and finish this, need to visualize the lr schedule to make sure it looks right
    return WarmupCosineSchedule(
        optimizer,
        warmup_steps,
        total_steps=total_steps,
        warmup_min_lr=warmup_min_lr,
        num_cycles=num_cycles,
        last_epoch=-1,
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


def create_multistep_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    multisteps: list[int],
    num_steps_per_epoch: int,
    gamma: float = 0.1,
    warmup_epochs: int = 0,
    warmup_lr_init: float = 0,
    t_in_epochs: bool = True,
):
    """Builds the multistep lr scheduler which reduces the learning rate by a factor of gamma at
    each milestone

    Args:
        optimizer: torch optimizer to update w/ the learning rate to be updated
        multisteps: list of epochs to reduce the learning rate at; if t_in_epochs is False then
                    these will be converted to steps
        gamma: multiplicative factor to reduce the learning rate by at each milestone
        warmup_epochs: number of epochs to linearly increase the learning rate over before decaying it
        warmup_lr_init: initial learning rate to start the linear warmup from
        t_in_epochs: if True, milestones and warmup_t are in epochs; if False, they are in steps
    """
    warmup_steps = int(warmup_epochs * num_steps_per_epoch)
    multi_steps = [i * num_steps_per_epoch for i in multisteps]

    return MultiStepLRScheduler(
        optimizer=optimizer,
        milestones=multi_steps,
        gamma=gamma,
        warmup_t=warmup_steps,
        warmup_lr_init=warmup_lr_init,
        t_in_epochs=t_in_epochs,
    )
