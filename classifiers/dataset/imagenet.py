from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as T


class ImageNet(datasets.ImageFolder):
    """ImageNet 2012 dataset from ILSVRC2012

    This class acts as a wrapper of torchvision.datasets.ImageFolder so we can include extra
    information and add a dev_mode.

    For more information on the dataset format, see the README
    """

    def __init__(
        self,
        image_folder: str,
        transforms: T,
        image_size: int = 224,
        dev_mode: bool = False,
    ):
        """TODO

        Args:
            image_folder: path to the images of the specific split
            dev_mode: only uses a small amount of samples to quickly run the code
        """
        super().__init__(root=image_folder, transform=transforms)

        self.image_size = image_size
        self.num_classes = 1000

        # Substantially reduces the dataset size to quickly test code
        # TODO: work on dev mode
        if dev_mode:
            self.samples = self.samples[:256]


def make_imagenet_transforms(dataset_split: str, img_size: int = 224):
    """Initialize transforms for the coco dataset

    These transforms are based on torchvision transforms but are overrided in data/transforms.py
    This allows for slight modifications in the the transform

    Args:
        dataset_split: which dataset split to use; `train` or `val`
        image_size: square size of the image to resize/crop to; default is 224
    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if dataset_split == "train":
        return T.Compose(
            [
                T.RandomResizedCrop(
                    img_size
                ),  # 224 is commonly used to pretrain classifiers
                T.RandomHorizontalFlip(p=0.5),
                normalize,
            ]
        )
    elif dataset_split == "val":
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(img_size),
                normalize,
            ]
        )
    # NOTE: ImageNet does not have a public test set
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def build_imagenet(
    root: str,
    dataset_split: str,
    image_size: int = 224,
    dev_mode: bool = False,
):
    """Initialize the ImageNet2012 dataset

    Args:
        root: full path to the dataset root
        split: which dataset split to use; `train` or `val`
        image_size: square size of the image to resize/crop to; default is 224
        dev_mode: whether to build the dataset in dev mode; if true, this only uses a few samples
                         to quickly run the code
    """
    imagenet_root = Path(root)

    if dataset_split == "train":
        images_dir = imagenet_root / "train"
    elif dataset_split == "val":
        images_dir = imagenet_root / "val"

    # Create the data augmentation transforms
    data_transforms = make_imagenet_transforms(dataset_split, img_size=image_size)

    dataset = ImageNet(
        image_folder=images_dir,
        transforms=data_transforms,
        dev_mode=dev_mode,
    )

    return dataset
