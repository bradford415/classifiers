from pathlib import Path

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data._utils.collate import default_collate


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


class MaskGenerator:
    """Mask Generator for SimMIM used to create binary masks that tell the model which patches
    of an image to hide
    """

    def __init__(
        self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6
    ):
        """Initalize the MaskGenerator

        Args:
            input_size: size of the input image after transformations; typically 192x192 or 224x224
            mask_patch_size: size of the patches to be masked; this is NOT related
                             to the Swin patch size e.g., Swin typically uses 4x4 patches;
                             must be divisible by input image size and model_patch_size must
                             be divisible by mask_patch_size
            model_patch_size: size of the patches used by the model; this this is typically
                              4x4 for Swin; must be divisible by mask_patch_size
            mask_ratio: the percentage of total patches to mask (e.g., 0.6 = 60% of patches are masked)
        """
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        # number of masked patches along one side; e.g., 192 // 32 = 6 → so the image is
        # divided into a 6×6 grid of mask patches.
        self.rand_size = self.input_size // self.mask_patch_size

        # number of patches in one mask patch; e.g., 32 // 4 = 8 ;
        # 4 => 4x4 patches used by swin during patchification
        self.scale = self.mask_patch_size // self.model_patch_size

        # number of masked patches for the entire image; assumes square input size
        self.token_count = self.rand_size**2

        # number of masked patches to actually mask
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        """Generates a binary mask indicating which patches to mask 1 = masked, 0 = visible

        Returns:
            mask: a 2D binary mask of shape (num_patches, num_patches) (1 = masked, 0 = visible)
                  where num_patches = input_size // model_patch_size;
                  e.g., input = 192x192, mask_patch_size = 32x32, patch_size = 4x4
                  then the returned mask will be 48x48
        """
        # randomly select indices of masked patches to mask (self.mask_count,)
        # self.token_count is the total number of patches in the image
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]

        # boolen mask where 1 indicates the patch is masked and 0 indicates it is visible
        # (self.token_count,)
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        # transform the boolean mask into masked patch grid (num_mask_patches, num_mask_patches)
        # and repeat across each axis for the number of patches in the each mask patch
        # to match the models patch resolution (num_patches, num_patches);
        # e.g., 4x4 patches, 32x32 masked patches -> 8 patches per mask patch -> 48x48 patch resolution
        #       which corresponds to 192x192 -> patchify (4x4) -> 48x48 patches so now each
        #       patch will have a boolean whether to mask or not
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    """Data augmentation transforms for SimMIM pretraining"""

    def __init__(
        self,
        dataset_split: str,
        img_size: int = 224,
        mask_patch_size: int = 32,
        model_patch_size: int = 4,
        mask_ratio: float = 0.6,
    ):
        """Initialize the SimMIM transforms

        Args:
            dataset_split: which dataset split to use; `train` or `val`
            img_size: square size of the image to resize/crop to
            mask_path_size: size of the patches to be masked; e.g., 32 means 32x32 pixels
            model_patch_size: size of the patches used by the model during patchification
            mask_ratio: the percentage of masked patches to us(e.g., 0.6 = 60% of patches are masked)
        """

        _normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        # TODO: determine if I need separate augmentations for train and val

        self.transform_img = T.Compose(
            [
                # random resize crop works as follows:
                #   1. first compute the new area by randomly choosing a scale
                #      between [67%, 100%] and multiply by the area of the original image
                #   2. then randomly pick an aspect ratio uniform(3/4, 4/3)
                #   3. from the new area and aspect ratio, compute the new height and width
                #      new_h = sqrt(new_area / aspect_ratio), new_w = aspect_ratio * new_h
                #      (formula is a little confusing but chatgpt can explain)
                #   4. randomly choose the top left corner for the cropped region
                #      y = Uniform(0, original_h - new_h)
                #      x = Uniform(0, original_w - new_w)
                #   5. finally resize the crop to img_size (224x224)
                # NOTE: `ratio` is the default parameter and `scale` is modified to match the simmim code
                T.RandomResizedCrop(
                    img_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),  # 224 is commonly used to pretrain classifiers
                T.RandomHorizontalFlip(),
                _normalize,
            ]
        )

        # create the mask generator which is a binary mask indicating which pixels to mask
        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

    def __call__(self, img):
        """Apply the simmim transforms to the input image and create the binary mask

        Args:
            img: TODO get type (pil?) original input image

        Returns:
            a tuple of:
                1. the transformed image tensor
                2. the binary mask tensor (1 = masked, 0 = visible) (num_pixels, num_pixels)
                TODO verify shape
        """
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask


def collate_fn_simmim(batch: list[tuple[]]):
    """Custom collate function to handle batches where each sample is a 
    tuple of ((image, mask), target)

    Args:
        batch: a list of samples for simmim where each sample is a tuple ((image, mask), target)
               - image: the transformed image tensor (c, h, w)
               - mask: the binary mask representing which patches to mask (1 = masked, 0 = visible)
                       (num_patches, num_patches)
                - target: the class label of the image (ignored during simmim pretraining)
    
    Returns:
        a 3 element list containing:
            1. a batch of images (b, c, h, w)
            2. a batch of masks (b, num_patches, num_patches)
            3. a batch of class labels (b,); again, I think this is ignored in simmim pretraining
    """
    if not isinstance(batch[0][0], tuple):
        # use the standard pytorch collate function if the first element is not a tuple
        return default_collate(batch)
    else:
        # simmmim case
        batch_num = len(batch)
        ret = []
        
        # loop through the image and mask tuple to create a two element list containing
        # a batch of images (b, c, h, w) and a batch of masks (b, num_patches, num_patches)
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(
                    default_collate([batch[i][0][item_idx] for i in range(batch_num)])
                )

        # create a batch of class labels for each image (b,)
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


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


def build_imagenet_simmim(
    root: str,
    dataset_split: str,
    image_size: int = 224,
    mask_patch_size: int = 32,
    model_patch_size: int = 4,
    mask_ratio: float = 0.6,
    dev_mode: bool = False,
):
    """Initialize the ImageNet2012 dataset for SimMIM pretraining

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
    data_transforms = SimMIMTransform(
        dataset_split,
        img_size=image_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio,
    )

    dataset = ImageNet(
        image_folder=images_dir,
        transforms=data_transforms,
        dev_mode=dev_mode,
    )

    return dataset
