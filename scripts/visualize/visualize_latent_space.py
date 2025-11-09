import datetime
from pathlib import Path

import torch
import yaml
from fire import Fire
from torch.utils.data import DataLoader

from classifiers.dataset.imagenet import build_imagenet_simmim, collate_fn_simmim
from classifiers.evaluate import load_model_checkpoint
from classifiers.models.create import create_simmim_model
from classifiers.visualize import plot_masked_patches


def visualize_simmim(
    model_config_path: str,
    model_weights_path: str,
    img_size: int = 192,
    num_images: int = 32,
    output_dir: str = "output",
):
    """Visualizes the masked patch predictions of the SimMIM pretrained model

    Args:
        model_config_path: path to the model configuration file that was used to create the SimMIM model
        model_path: path to the weights file of the trained SimMIM model
    """
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    output_path = (
        Path(output_dir)
        / "visualizations"
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"saving outputs to {str(output_path)}")

    backbone_name = model_config["backbone"]
    backbone_params = model_config["params"]
    model = create_simmim_model(
        backbone_name=backbone_name,
        backbone_args=backbone_params,
        # NOTE: num_classes is hardcoded as 0 since there's no classification head in simmim
        image_size=img_size,
    )
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the model weights
    load_model_checkpoint(model_weights_path, model=model, device=device)

    dataset_root_path = "/mnt/d/datasets/imagenet"
    dataset_train = build_imagenet_simmim(
        root=dataset_root_path, dataset_split="val", image_size=img_size
    )
    dataloader_val = DataLoader(
        dataset_train,
        drop_last=True,
        collate_fn=collate_fn_simmim,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # save the iterable object or else the dataloader will restart from the beginning each time
    dataloader_iter = iter(dataloader_val)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    all_images = []
    all_masks = []
    all_predicted_pixels = []
    for _ in range(num_images):
        images, masks, _ = next(dataloader_iter)
        images = images.to(device)
        masks = masks.to(device)

        with torch.inference_mode():
            _, predicted_pixels = model(images, masks)

        all_images.append(images)
        all_masks.append(masks)
        all_predicted_pixels.append(predicted_pixels)

    all_images = torch.concat(all_images, dim=0)
    all_masks = torch.concat(all_masks, dim=0)
    all_predicted_pixels = torch.concat(all_predicted_pixels, dim=0)
    plot_masked_patches(
        all_images.cpu(),
        all_masks.cpu(),
        all_predicted_pixels.cpu(),
        patch_size=model_config["params"]["patch_size"],
        save_dir=str(output_path),
    )


if __name__ == "__main__":
    Fire(visualize_simmim)
