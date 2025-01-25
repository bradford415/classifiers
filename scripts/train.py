import datetime
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from classifiers.data.imagenet import build_imagenet
from classifiers.evaluate import topk_accuracy
from classifiers.models import darknet53, vit
from classifiers.trainer import Trainer
from classifiers.utils import reproduce, schedulers

classifier_map: Dict[str, Any] = {"vit": vit}

dataset_map: Dict[str, Any] = {"ImageNet": build_imagenet}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,
    "cosine_annealing": schedulers.make_cosine_anneal,  # quickly decays lr then spikes back up for a "warm restart" https://paperswithcode.com/method/cosine-annealing
}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    dev_mode = base_config["dev_mode"]

    # Initialize paths
    output_path = (
        Path(base_config["output_dir"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    if dev_mode:
        log.info("NOTE: executing in dev mode")
        base_config["train"]["batch_size"] = 2
        base_config["validation"]["batch_size"] = 2

    log.info("initializing...\n")
    log.info("writing outputs to %s", str(output_path))

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Set gpu parameters
    train_kwargs = {
        "batch_size": base_config["train"]["batch_size"],
        "shuffle": True,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }

    # Set device specific characteristics
    use_cpu = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))
    elif torch.mps.is_available():
        base_config["dataset"]["root"] = base_config["dataset"]["root_mac"]
        device = torch.device("mps")
        log.info("Using: %s", device)
    else:
        use_cpu = True
        device = torch.device("cpu")
        log.info("Using CPU")

    if not use_cpu:
        gpu_kwargs = {
            "pin_memory": True,
        }

        train_kwargs.update(gpu_kwargs)
        val_kwargs.update(gpu_kwargs)

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", dev_mode=dev_mode, **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=dev_mode, **dataset_kwargs
    )

    dataloader_train = DataLoader(
        dataset_train,
        drop_last=True,
        **train_kwargs,
    )
    dataloader_val = DataLoader(
        dataset_val,
        drop_last=True,
        **val_kwargs,
    )

    # Initalize the classifier and extract initialization the parameters
    classifier_name = model_config["classifier"]["name"]
    classifier_params = model_config[classifier_name]
    model = classifier_map[classifier_name](**classifier_params)

    # Compute and log the number of params in the model
    reproduce.model_info(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    log.info("\nclassifier: %s", model_config["classifier"]["name"])

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract the learning parameters such as lr, optimizer params and lr scheduler
    learning_config = train_args["learning_config"]
    learning_params = base_config[learning_config]

    # Initialize training objects
    optimizer, lr_scheduler = _init_training_objects(
        model_params=model.parameters(),
        optimizer=learning_params["optimizer"],
        scheduler=learning_params["lr_scheduler"],
        learning_rate=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"],
    )

    trainer = Trainer(
        output_dir=str(output_path),
        device=device,
        log_train_steps=base_config["log_train_steps"],
    )

    # Save configuration files for reproducibility
    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.json", "model_config.json"],
        output_path=output_path / "reproduce",
    )

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "class_names": dataset_train.class_names,
        "start_epoch": train_args["start_epoch"],
        "epochs": train_args["epochs"],
        "ckpt_epochs": train_args["ckpt_epochs"],
        "checkpoint_path": train_args["checkpoint_path"],
    }
    trainer.train(**trainer_args)


def _init_training_objects(
    model_params: Iterable,
    optimizer: str = "sgd",
    scheduler: str = "step_lr",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: Iterable[float, float] = (0.9, 0.999),
    lr_drop: int = 200,
):
    optimizer = optimizer_map[optimizer](
        model_params, lr=learning_rate, weight_decay=weight_decay, betas=betas
    )
    lr_scheduler = scheduler_map[scheduler](optimizer)

    return optimizer, lr_scheduler


if __name__ == "__main__":
    Fire(main)
