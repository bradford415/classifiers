from .config import *
from .schedulers import *

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,
    "cosine_annealing": make_cosine_anneal,  # quickly decays lr then spikes back up for a "warm restart" https://paperswithcode.com/method/cosine-annealing
    "warmup_cosine_decay": warmup_cosine_decay,  # linearly increases lr then decays it with a cosine function
}


solver_configs = {"resnet50_imagenet": resnet50_imagenet_config}
