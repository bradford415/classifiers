import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from classifiers.dataset.transforms import Unnormalize

matplotlib.use("Agg")


def plot_masked_patches(images: torch.Tensor, masks: torch.Tensor, predicted_pixels: torch.Tensor, save_dir: str, patch_size: int = 4):
    
    # TODO: plot the fully predicted image at the end, not just masked version, like in swin figure 4&5
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # repeat the mask by patch size to match the image size
    masks = masks.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    assert masks.shape[1:] == images.shape[2:], f"mask should be the same shape as images, check the patch size"
    
    untransformed_imgs = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images) * 255
    masked_images = untransformed_imgs * (1 - masks.unsqueeze(1))
    predicted_masks = masked_images + (predicted_pixels * masks.unsqueeze(1))

    untransformed_imgs = untransformed_imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    masked_images = masked_images.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    predicted_masks = predicted_masks.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    untransformed_imgs = untransformed_imgs[:32]
    masked_images = masked_images[:32]
    predicted_masks = predicted_masks[:32]
    
    imgs_per_plot = 4
    
    num_plots = math.ceil(untransformed_imgs.shape[0] / imgs_per_plot)
    for plt_i in range(num_plots):
        # determine how many images to plot
        imgs = untransformed_imgs[plt_i*imgs_per_plot:(plt_i+1)*imgs_per_plot]
        mask_imgs = masked_images[plt_i*imgs_per_plot:(plt_i+1)*imgs_per_plot]
        pred_masks = predicted_masks[plt_i*imgs_per_plot:(plt_i+1)*imgs_per_plot]
        
        num_figs = imgs.shape[0]
        
        fig, axs = plt.subplots(num_figs, 3, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # tight spacing
        axs[0, 0].set_title("original", fontsize=14)
        axs[0, 1].set_title("masked", fontsize=14)
        axs[0, 2].set_title("predicted", fontsize=14)

        for idx, (img, mask_img, pred_mask) in enumerate(zip(imgs, mask_imgs, pred_masks)):
            
            axs[idx, 0].imshow(img)
            axs[idx, 0].axis("off")
            axs[idx, 1].imshow(mask_img)
            axs[idx, 1].axis("off")
            axs[idx, 2].imshow(pred_mask)
            axs[idx, 2].axis("off")
        
        fig.savefig(Path(save_dir) / f"masked_patches_{plt_i}_original.jpg", bbox_inches="tight")
    

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
