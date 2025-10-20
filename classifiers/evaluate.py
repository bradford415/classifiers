import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader_test: Iterable,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tuple, List]:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        criterion: Loss function; only used to inspect the loss on the val set,
                    not used for ba ropagation
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on

    Returns:
        A Tuple containing
            1. A Tuple of the (prec, rec, ap, f1, and class) per class
            2. A list of tuples containing the image_path and detections after postprocessing with nms

    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for steps, (samples, targets) in enumerate(
        tqdm(dataloader_test, desc="Evaluating", ncols=100)
    ):
        samples = samples.to(device)
        targets = targets.to(device)

        # (b, num_classes)
        preds = model(samples)

        loss = criterion(preds, targets)

        acc1, acc5 = topk_accuracy(preds, targets, topk=(1, 5))
        losses.update(loss.item(), samples.shape[0])
        top1.update(acc1[0], samples.shape[0])
        top5.update(acc5[0], samples.shape[0])

    log.info("top 1 accuracy of %.2f%%", top1.avg.item())

    return losses, top1


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module = None,
    optimizer: nn.Module = None,
    device=torch.device("cpu"),
    lr_scheduler: Optional[nn.Module] = None,
):
    """Load the checkpoints of a trained or pretrained model from the state_dict file;
    this could be from a fully trained model or a partially trained model that you want
    to resume training from.

    Args:
        checkpoint_path: path to the weights file to resume training from
        model: the model being trained
        optimizer: the optimizer used during training
        device: the device to map the checkpoints to
        lr_scheduler: the learning rate scheduler used during training

    Returns:
        the epoch to resume training on
    """
    # TODO: consider making this a class method of the model `from_pretrained`
    # Load the torch weights
    weights = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # load the state dictionaries for the necessary training modules
    if model is not None:
        missing_keys, _ = model.load_state_dict(weights["model"])
    if optimizer is not None:
        optimizer.load_state_dict(weights["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(weights["lr_scheduler"])

    start_epoch = 1
    if "epoch" in weights:
        # should only be False if loading a model not trained with this repo
        start_epoch = weights["epoch"]

    if not missing_keys:
        print("\nAll keys matched the model when loading its state dict")
    else:
        print(
            f"\nNot all keys matched the model when loading the state dict; missing keys: {missing_keys}"
        )

    return start_epoch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero out all the values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with the new value

        Args:
            val: new value to add
            n: weight of the new value (e.g., batch size); note that
                the sum is multiplied by `n`, this is because the loss
                is typically averged over the batch in the loss function so when we
                go to compute the average loss for the entire epoch we need to weight
                the loss by the batch size (essentially undoes the average in the loss function)
        """
        self.val = val
        self.sum += (
            val * n
        )  # multiply by n to get the total loss over the batch (undo the average in the loss function)
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output, target, topk: Iterable = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k;
    multplies by 100 to get the percentage

    Args:
        TODO
        topk: iterable of k values to compute the top k accuracy over
    """
    # TODO: comment this function
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
