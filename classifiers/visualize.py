import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from classifiers.dataset.transforms import Unnormalize

matplotlib.use("Agg")


def plot_masked_patches(
    images: torch.Tensor,
    masks: torch.Tensor,
    predicted_pixels: torch.Tensor,
    save_dir: str,
    patch_size: int = 4,
):
    """Visualize the masked patch predictions from simmim pretraining

    NOTE: for visualization it's very important to add the normalized predictions
          before undoing the normalization or else they will be on the wrong scale
          and look off

    Args:
        images: the raw input images (after transforms) that is passed into the simmim model(b, c, h, w)
        masks: a binary mask representing which patches to mask (b, h / patch_size, w / patch_size)
        predicted_pixels: the predicted pixels from simmim (b, c, h, w)
    """
    # TODO: plot the fully predicted image at the end, not just masked version, like in swin figure 4&5
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # repeat the mask by patch size to match the image size
    masks = masks.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    assert (
        masks.shape[1:] == images.shape[2:]
    ), f"mask should be the same shape as images, check the patch size"

    # create a masked image and its patch predictions
    masked_images = images * (1 - masks.unsqueeze(1))
    predicted_masks = masked_images + (predicted_pixels * masks.unsqueeze(1))

    unnormalize_transform = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # unnormalize
    untransformed_imgs = unnormalize_transform(images) * 255
    untransformed_masked_imgs = unnormalize_transform(masked_images) * 255
    untransformed_pred_masks = unnormalize_transform(predicted_masks) * 255
    untransformed_all_preds = unnormalize_transform(predicted_pixels) * 255

    untransformed_imgs = untransformed_imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    masked_images = (
        untransformed_masked_imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    )
    predicted_masks = (
        untransformed_pred_masks.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    )
    all_pred_pixels = (
        untransformed_all_preds.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    )

    untransformed_imgs = untransformed_imgs[:32]
    masked_images = masked_images[:32]
    predicted_masks = predicted_masks[:32]

    imgs_per_plot = 4

    num_plots = math.ceil(untransformed_imgs.shape[0] / imgs_per_plot)
    for plt_i in range(num_plots):
        # determine how many images to plot
        imgs = untransformed_imgs[plt_i * imgs_per_plot : (plt_i + 1) * imgs_per_plot]
        mask_imgs = masked_images[plt_i * imgs_per_plot : (plt_i + 1) * imgs_per_plot]
        pred_masks = predicted_masks[
            plt_i * imgs_per_plot : (plt_i + 1) * imgs_per_plot
        ]
        all_pred_pixs = all_pred_pixels[
            plt_i * imgs_per_plot : (plt_i + 1) * imgs_per_plot
        ]

        num_figs = imgs.shape[0]

        fig, axs = plt.subplots(num_figs, 4, figsize=(10, 8))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # tight spacing
        axs[0, 0].set_title("original", fontsize=14)
        axs[0, 1].set_title("masked", fontsize=14)
        axs[0, 2].set_title("predicted masks", fontsize=14)
        axs[0, 3].set_title("all predicted pixels", fontsize=14)

        for idx, (img, mask_img, pred_mask, pred_pixs) in enumerate(
            zip(imgs, mask_imgs, pred_masks, all_pred_pixs)
        ):

            axs[idx, 0].imshow(img)
            axs[idx, 0].axis("off")
            axs[idx, 1].imshow(mask_img)
            axs[idx, 1].axis("off")
            axs[idx, 2].imshow(pred_mask)
            axs[idx, 2].axis("off")
            axs[idx, 3].imshow(pred_pixs)
            axs[idx, 3].axis("off")

        fig.savefig(
            Path(save_dir) / f"masked_patches_{plt_i}_original.jpg", bbox_inches="tight"
        )


def plot_loss(
    train_loss: list[float], save_dir: str, val_loss: Optional[list[float]] = None
):
    """Plots the total train and optionally val loss per epoch"""
    save_name = Path(save_dir) / "total_loss.jpg"

    x = np.arange(len(train_loss)) + 1
    fig, ax = plt.subplots(1)

    legend_vals = ["train loss"]
    ax.plot(x, train_loss)

    if val_loss is not None:
        legend_vals.append("val loss")
        ax.plot(x, val_loss)

    plt.legend(legend_vals)
    plt.title("total loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_lr(lr: list[float], x_label, save_dir: str, epoch: Optional[int] = None):
    """Plots the learning rate per step; this is useful to visualize what the scheduler is doing"""
    save_name = Path(save_dir) / "learning_rate.jpg"

    x = np.arange(len(lr)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, lr)

    plt.legend(["learning rate"])
    plt.title("lr per step")
    ax.set_xlabel(x_label)
    ax.set_ylabel("learning rate")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_acc1(val_acc1: list[float], save_dir: str):
    """Plots the validation top 1 accuracy per epoch"""
    save_name = Path(save_dir) / "acc1_curve.jpg"

    x = np.arange(len(val_acc1)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, val_acc1)

    plt.title("validation top 1 acc per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("top 1 acc")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()
