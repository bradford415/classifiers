from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


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


def plot_lr(lr: list[float], x_label, save_dir: str):
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
