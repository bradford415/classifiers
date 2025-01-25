import torch


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
