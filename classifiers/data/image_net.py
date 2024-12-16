# Dataset class for the ImageNet dataset
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms as T


def _make_imagenet_transforms(dataset_split):
    """Initialize transforms for the coco dataset

    These transforms are based on torchvision transforms but are overrided in data/transforms.py
    This allows for slight modifications in the the transform

    Args:
        dataset_split: which dataset split to use; `train` or `val`
    """

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if dataset_split == "train":
        return T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                normalize,
            ]
        )
    elif dataset_split == "val":
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                normalize,
            ]
        )
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def build_imagenet(
    root: str,
    dataset_split: str,
    dev_mode: bool = False,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        split: which dataset split to use; `train` or `val`
        dev_mode: Whether to build the dataset in dev mode; if true, this only uses a few samples
                         to quickly run the code
    """
    coco_root = Path(root)

    data_transforms = _make_imagenet_transforms(dataset_split)

    # TODO
    dataset = datasets.ImageFolder(
        image_folder=images_dir,
        transforms=data_transforms,
        dev_mode=dev_mode,
        split=dataset_split,
    )

    return dataset
