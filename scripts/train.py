import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from classifiers.dataset.imagenet import build_imagenet
from classifiers.models import create_vit, resnet50
from classifiers.solvers import solver_configs  # TODO: decouple from this
from classifiers.solvers.build import build_solvers
from classifiers.trainer import Trainer
from classifiers.utils import reproduce

classifier_map: Dict[str, Any] = {
    "vit-b16": create_vit,
    "resnet50": resnet50,
}

dataset_map: Dict[str, Any] = {"ImageNet": build_imagenet}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: path to the detection model configuration file

    """


def main(
    base_config_path: str = "configs/train-imagenet-vit.yaml",
    model_config_path: str = "configs/vit/vit-base-16.yaml",
    dataset_root: Optional[str] = None,
    backbone_weights: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired training configuration file; by default the train-imagenet-vit.yaml file is used which
                          trains from scratch (i.e., no pretrained backbone weights)
        model_config_path: path to the classifier model configuration file; by default the ViT-B/16 model
        dataset_root: path to the the root directory of the dataset
        checkpoint_path: path to the weights of the classifier model; this can be used to resume training or inference;
    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Override configuration parameters if CLI arguments are provided; this allows external users
    # to easily run the project without messing with the configuration files
    if dataset_root is not None:
        base_config["dataset"]["root"] = dataset_root
    if checkpoint_path is not None:
        base_config["train"]["checkpoint_path"] = checkpoint_path

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

    log.info("initializing...\n")
    log.info("writing outputs to %s", str(output_path))

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Extract solver config parameters; TODO: decouple from solver
    solver_config = solver_configs[base_config["train"]["solver_config"]]()

    # Save configuration files and parameters
    reproduce.save_configs(
        config_dicts=[
            (base_config, "base_config.yaml"),
            (model_config, "model_config.yaml"),
        ],
        solver_dict=(solver_config.to_dict(), "solver_config.json"),
        output_path=output_path / "reproduce",
    )

    # Extract the train arguments from base config
    train_args = base_config["train"]

    if dev_mode:
        log.info("NOTE: executing in dev mode")
        solver_config.training.batch_size = 2
        solver_config.validation.batch_size = 2

    # Extract training and val params
    batch_size = train_args["batch_size"]
    effective_bs = train_args["effective_batch_size"]
    epochs = train_args["epochs"]

    val_batch_size = train_args["batch_size"]

    image_size = base_config["dataset"]["image_size"]

    # Set gpu parameters
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }
    val_kwargs = {
        "batch_size": val_batch_size,
        "shuffle": False,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }

    # Gradient accumulation;
    # accumulate (sum) gradients for effective//batch_size steps before updating the weights;
    # therefore batch_size must be divisible by grad_accum_bs; beneficial with constrained memory when
    # trying to replicate a larger batch size
    if effective_bs % batch_size == 0 and effective_bs >= batch_size:
        grad_accum_steps = effective_bs // batch_size
    else:
        raise ValueError(
            "grad_accum_bs must be divisible by batch_size and greater than or equal to batch_size"
        )

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

    dataset_kwargs = {"root": base_config["dataset"]["root"], "image_size": image_size}
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

    log.info("\nusing image size %d\n", image_size)

    # Extract initialization parameters for the classifier; TODO create a create classifier function
    classifier_name = model_config["classifier"]["name"]
    if "vit" in classifier_name:
        classifier_params = {
            "image_size": image_size,
            "num_classes": dataset_train.num_classes,
            **model_config["params"],
        }
    elif "resnet" in classifier_name:
        classifier_params = {
            "num_classes": dataset_train.num_classes,
            **model_config["params"],
        }
    else:
        ValueError("classifier not recognized.")

    # Initalize classifier
    model = classifier_map[classifier_name](**classifier_params)

    # Compute and log the number of params in the model
    reproduce.model_info(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    log.info("\nclassifier: %s", model_config["classifier"]["name"])

    optimizer, lr_scheduler = build_solvers(
        model.parameters(), solver_config.optimizer, solver_config.lr_scheduler
    )

    total_steps = (len(dataloader_train) * epochs) // grad_accum_steps
    log.info(
        "total effective steps (steps * epochs // grad_accum_steps): %d", total_steps
    )

    trainer = Trainer(
        output_dir=str(output_path),
        step_lr_on=solver_config["step_lr_on"],
        device=device,
        log_train_steps=base_config["log_train_steps"],
        disable_amp=base_config["disable_amp"],
    )

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "grad_accum_steps": grad_accum_steps,
        "max_norm": train_args["max_norm"],
        "start_epoch": 1,
        "epochs": epochs,
        "ckpt_epochs": train_args["ckpt_epochs"],
        "checkpoint_path": train_args["checkpoint_path"],
    }
    trainer.train(**trainer_args)


if __name__ == "__main__":
    Fire(main)
