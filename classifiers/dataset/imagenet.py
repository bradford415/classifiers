from pathlib import Path

import numpy as np
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
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        """Initalize the MaskGenerator
        
        Args:
            input_size: size of the input image after transformations
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
        
        # number of patches in one mask patch; e.g., 32 // 4 = 8
        self.scale = self.mask_patch_size // self.model_patch_size
        
        # number of masked patches for the entire image; assumes square input size
        self.token_count = self.rand_size ** 2
        
        # number of masked patches to actually mask 
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        ### start here  
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    
    
class SimMIMTransform:
    def __init__(self, dataset_split: str, img_size: int = 224):
        
        _normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        
        # TODO: determine if I need separate augmentations for train and val
        
        self.transform_img = T.Compose([
                # random resize crop works as so:
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
                # NOTE: `ratio`` is the default parameter and `scale`` is modified to match the simmim code
                T.RandomResizedCrop(
                    img_size, scale=(0.67, 1.), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),  # 224 is commonly used to pretrain classifiers
            T.RandomHorizontalFlip(),
            _normalize
        ]) 
        
        
        #### start here go through MaskGenerator
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


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
    data_transforms = make_imagenet_transforms_simmim(dataset_split, img_size=image_size)

    dataset = ImageNet(
        image_folder=images_dir,
        transforms=data_transforms,
        dev_mode=dev_mode,
    )

    return dataset
