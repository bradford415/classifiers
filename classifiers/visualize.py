from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss: list[float], val_loss: list[float], save_dir: str):
    """Plots the total val and train loss per epoch"""
    save_name = Path(save_dir) / "total_loss.jpg"

    x = np.arange(len(train_loss)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, train_loss)
    x.plot(x, val_loss)

    plt.legend(["train loss", "val loss"])
    plt.title("total loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

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
    ax.set_ylabel("mAP")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()

