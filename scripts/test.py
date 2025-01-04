import datetime
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from classifiers.data.imagenet import build_coco
from classifiers.data.collate_functions import collate_fn
from classifiers.evaluate import evaluate, load_model_checkpoint
from classifiers.models import Yolov3, Yolov4
from classifiers.models.backbones import backbone_map
from classifiers.models.darknet import Darknet
from classifiers.utils import reproduce
from classifiers.visualize import plot_all_detections

# TODO: should move this to its own file
detectors_map: Dict[str, Any] = {"yolov3": Yolov3, "yolov4": Yolov4}

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired base configuration file
        model_config_path: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize paths
    output_path = (
        Path(base_config["output_dir"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "testing.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("initializing...")
    log.info("outputs beings saved to %s\n", str(output_path))

    # apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_kwargs = {
        "batch_size": base_config["test"]["batch_size"],
        "shuffle": False,
    }

    if use_cuda:
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))

        cuda_kwargs = {
            "num_workers": base_config["dataset"]["num_workers"],
            "pin_memory": True,
        }

        test_kwargs.update(cuda_kwargs)
    else:
        log.info("Using CPU")

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_test = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=base_config["dev_mode"], **dataset_kwargs
    )

    dataloader_test = DataLoader(
        dataset_test,
        collate_fn=collate_fn,
        drop_last=True,
    )

    anchors = model_config["priors"]["anchors"]

    # number of anchors per scale
    num_anchors = len(anchors[0])

    # strip away the outer list to make it a 2d python list
    anchors = [anchor for anchor_scale in anchors for anchor in anchor_scale]

    # Initalize the detector backbone; typically some feature extractor
    backbone_name = model_config["backbone"]["name"]
    if backbone_name in model_config["backbone"]:
        backbone_params = model_config["backbone"][backbone_name]
    else:
        backbone_params = {}
    backbone = backbone_map[backbone_name](**backbone_params)

    # detector args
    model_components = {
        "backbone": backbone,
        "num_classes": dataset_test.num_classes,
        "anchors": anchors,
    }

    # Initialize detection model and load its state_dict
    model = detectors_map[model_config["detector"]](**model_components)
    model.to(device)

    start_epoch = load_model_checkpoint(base_config["test"]["checkpoint_path"], model)

    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.json", "model_config.json"],
        output_path=output_path / "reproduce",
    )
    # Build trainer args used for the training
    evaluation_args = {
        "output_path": output_path,
        "device": device,
    }
    batch_metrics, image_detections = evaluate(
        model, dataloader_test, dataset_test.class_names, **evaluation_args
    )

    save_dir = output_path / "test"

    if base_config["plot_detections"]:
        plot_all_detections(
            image_detections, classes=dataset_test.class_names, output_dir=save_dir
        )


if __name__ == "__main__":
    Fire(main)
