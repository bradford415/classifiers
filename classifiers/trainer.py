import datetime
import logging
import time
from abc import ABC
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from classifiers.evaluate import (
    AverageMeter,
    evaluate,
    load_model_checkpoint,
    topk_accuracy,
)
from classifiers.visualize import plot_acc1, plot_loss, plot_lr

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Trainer TODO: comment"""

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        step_lr_on: str,
        device: torch.device = torch.device("cpu"),
        log_train_steps: int = 20,
        amp_dtype: str = "float16",
        disable_amp: bool = False,
    ):
        """Constructor for the Trainer class

        Args:
            model: pytorch model to train
            output_path: Path to save the train outputs
            use_cuda: Whether to use the GPU
        """
        self.model = model

        if step_lr_on not in {"epochs", "steps"}:
            raise ValueError("step_lr_on must be either 'epochs' or 'steps'")

        self.device = device

        self.output_dir = Path(output_dir)
        self.log_train_steps = log_train_steps

        self.step_lr_on = step_lr_on

        if amp_dtype == "float16":
            self.amp_dtype = torch.float16
        elif amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError("amp_dtype must be either 'float16' or 'bfloat16'")

        if disable_amp:
            self.enable_amp = False
        else:
            # mixed precision training not yet supported on mps
            self.enable_amp = True if not self.device.type == "mps" else False

        # Metrics
        self.learning_rate = []

        # NOTE: for the ViT model, mixed precision training created only NaNs, disabling for now;
        #       seems to be a long standing bug: https://github.com/pytorch/pytorch/issues/40497
        # self.enable_amp = False

    def train(
        self,
        criterion: nn.Module,
        dataloader_train: data.DataLoader,
        dataloader_val: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        grad_accum_steps: int = 1,
        max_norm: float = 1.0,
        start_epoch: int = 1,
        epochs: int = 100,
        ckpt_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
    ):
        """Trains a model

        Specifically, this method trains a model for n epochs and evaluates on the validation set.
        A model checkpoint is saved at user-specified intervals.

        Args:
            model: pytorch model to be trained
            criterion: the loss function to use for training
            dataloader_train: torch dataloader to loop through the train dataset
            dataloader_val: torch dataloader to loop through the val dataset
            optimizer: optimizer which determines how to update the weights
            scheduler: scheduler which determines how to change the learning rate
            grad_accum_steps: TODO
            max_norm: TODO
            start_epoch: epoch to start the training on; starting at 1 is a good default because it makes
                         checkpointing and calculations more intuitive
            epochs: the epoch to end training on; unless starting from a check point, this will be the number of epochs to train for
            ckpt_every: save the model after n epochs
        """
        log.info("\ntraining started\n")

        if checkpoint_path is not None:
            start_epoch = load_model_checkpoint(
                checkpoint_path, self.model, optimizer, self.device, scheduler
            )
            log.info(
                "NOTE: A checkpoint file was provided, the model will resume training at epoch %d",
                start_epoch,
            )

        if self.enable_amp:
            log.info("using mixed precision training")
            scaler = torch.amp.GradScaler(self.device.type)
        else:
            scaler = None

        total_train_start_time = time.time()

        last_best_path = None

        # NOTE: the dataloaders can be visualized with scripts/visualizations/dataloaders.py

        best_acc = 0.0
        train_loss = []
        val_loss = []
        lr_vals = []
        epoch_acc1 = []
        for epoch in range(start_epoch, epochs + 1):
            self.model.train()

            # Track the time it takes for one epoch (train and val)
            one_epoch_start_time = time.time()

            # Train one epoch
            train_loss_meter = self._train_one_epoch(
                criterion,
                dataloader_train,
                optimizer,
                scheduler,
                epoch,
                grad_accum_steps,
                max_norm,
                scaler,
            )

            train_loss.append(train_loss_meter.avg)

            # Evaluate the model on the validation set
            log.info("\nEvaluating on validation set — epoch %d", epoch)

            val_loss_meter, acc1_meter = self._evaluate(criterion, dataloader_val)
            val_loss.append(val_loss_meter.avg)

            if self.step_lr_on == "epochs":
                curr_lr = round(optimizer.state_dict()["param_groups"][0]["lr"], 8)
                self.learning_rate.append(curr_lr)

            # Increment lr scheduler every epoch
            if scheduler is not None and self.step_lr_on == "epochs":
                if not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                else:
                    scheduler.step(val_loss[-1])

            acc1 = acc1_meter.avg.item()
            epoch_acc1.append(acc1)

            if acc1 > best_acc:
                best_acc = acc1

                acc_str = f"{acc1:.2f}".replace(".", "-")
                best_path = self.output_dir / "checkpoints" / f"best_acc1_{acc_str}.pt"
                best_path.parents[0].mkdir(parents=True, exist_ok=True)

                log.info(
                    "new best top 1 accuracy of %.2f found at epoch %d; saving checkpoint",
                    acc1,
                    epoch,
                )
                self.save_model(
                    optimizer, epoch, save_path=best_path, lr_scheduler=scheduler
                )

                # delete the previous best model's checkpoint to save space
                if last_best_path is not None:
                    last_best_path.unlink(missing_ok=True)
                last_best_path = best_path

            plot_loss(train_loss, val_loss, save_dir=str(self.output_dir))
            plot_acc1(epoch_acc1, save_dir=str(self.output_dir))

            # Create csv file of training stats per epoch
            train_dict = {
                "epoch": list(np.arange(start_epoch, epoch + 1)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "acc1": epoch_acc1,
            }
            pd.DataFrame(train_dict).round(5).to_csv(
                self.output_dir / "train_stats.csv", index=False
            )

            # Save the model every ckpt_epochs
            if (epoch) % ckpt_epochs == 0:
                ckpt_path = self.output_dir / "checkpoints" / f"checkpoint{epoch:04}.pt"
                ckpt_path.parents[0].mkdir(parents=True, exist_ok=True)
                self._save_model(
                    self.model,
                    optimizer,
                    epoch,
                    save_path=ckpt_path,
                    lr_scheduler=scheduler,
                )

            # Save and overwrite the checkpoint with the highest top 1 accuracy
            if round(acc1, 4) > round(best_acc, 4):
                best_acc = acc1

                acc1_str = f"{acc1*100:.2f}".replace(".", "-")
                best_path = self.output_dir / "checkpoints" / f"best_acc_{acc1_str}.pt"
                best_path.parents[0].mkdir(parents=True, exist_ok=True)

                log.info(
                    "new best top 1 accuracy of %.2f found at epoch %d; saving checkpoint",
                    acc1 * 100,
                    epoch,
                )
                self._save_model(
                    self.model,
                    optimizer,
                    epoch,
                    save_path=best_path,
                    lr_scheduler=scheduler,
                )

                # delete the previous best accuracy model's checkpoint
                if last_best_path is not None:
                    last_best_path.unlink(missing_ok=True)
                last_best_path = best_path

            # Current epoch time (train/val)
            one_epoch_time = time.time() - one_epoch_start_time
            one_epoch_time_str = str(datetime.timedelta(seconds=int(one_epoch_time)))
            log.info("\nEpoch time  (h:mm:ss): %s", one_epoch_time_str)

        # Entire training time
        total_time = time.time() - total_train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info(
            "Training time for %d epochs (h:mm:ss): %s ",
            start_epoch - epochs,
            total_time_str,
        )

    def _train_one_epoch(
        self,
        criterion: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        grad_accum_steps: int,
        max_norm: Optional[float],
        scaler: torch.amp,
    ):
        """Train one epoch

        Args:
            criterion: loss function
            dataloader_train: dataloader for the training set
            optimizer: optimizer to update the models weights
            scheduler: learning rate scheduler to update the learning rate
            epoch: current epoch; used for logging purposes
            scaler: scaler for mixed precision
        """
        # TODO: should probably move these metric functins/classes in evaluate to its own metrics.py file
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        epoch_loss = []
        epoch_lr = []  # Store lr every epoch so I can visualize the scheduler
        curr_lr = None
        for steps, batch in enumerate(dataloader_train, 1):

            samples, targets, preds, loss = self.train_step(batch, criterion, grad_accum_steps)

            acc1, acc5 = topk_accuracy(preds, targets, topk=(1, 5))
            losses.update(
                loss.item(), samples.shape[0]
            )  # TODO: Need to check if this is accurate now that I added gradient accumulation; I think it is
            top1.update(acc1[0], samples.shape[0])
            top5.update(acc5[0], samples.shape[0])

            # https://github.com/jeonsworld/ViT-pytorch/blob/460a162767de1722a014ed2261463dbbc01196b6/train.py#L198
            # breakpoint()

            # Calculate and accumulate gradients
            if self.enable_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if steps % grad_accum_steps == 0:
                if self.enable_amp:
                    # Unscales the gradients of optimizer's assigned params in-place and clip as usual
                    if max_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    optimizer.step()

                optimizer.zero_grad()

                curr_lr = round(optimizer.state_dict()["param_groups"][0]["lr"], 8)

                if self.step_lr_on == "steps":
                    # I think having this in the grad_accum_steps if block is correct since the lr is only used during
                    # optimizer step
                    self.learning_rate.append(curr_lr)

                # Increment lr scheduler every effective batch_size (grad_accum_steps)
                if scheduler is not None and self.step_lr_on == "steps":
                    if not isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step()
                    else:
                        scheduler.step(loss.item())

            epoch_loss.append(loss.item())

            if (steps) % 10 == 0:
                # Update the learning rate plot after n steps
                plot_lr(
                    self.learning_rate,
                    x_label=self.step_lr_on,
                    save_dir=str(self.output_dir),
                )

            if (steps) % self.log_train_steps == 0:
                curr_lr = optimizer.param_groups[0]["lr"]
                log.info(
                    "epoch: %-10d iter: %d/%-12d train_loss: %-10.4f curr_lr: %-12.6f",  # -n = right padding
                    epoch,
                    steps,
                    len(dataloader_train),
                    # NOTE: need to multiply by grad_accum_steps because the loss variable is scaled
                    # and overwritten every step; the gradients is what stores the accumulated information
                    loss.item() * grad_accum_steps,
                    curr_lr,
                )

        return losses

    @torch.no_grad()
    def _evaluate(
        self,
        criterion: nn.Module,
        dataloader_val: Iterable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A single forward pass to evaluate the val set after training an epoch

        Args:
            criterion: Loss function; only used to inspect the loss on the val set,
                       not used for backpropagation
            dataloader_val: Dataloader for the validation set
            device: Device to run the model on

        Returns:
            A Tuple of the (prec, rec, ap, f1, and class) per class
        """

        # NOTE: evaluate
        loss, top1 = evaluate(
            self.model,
            dataloader_val,
            criterion,
            device=self.device,
        )

        return loss, top1

    def _save_model(
        self,
        optimizer,
        current_epoch,
        save_path,
        lr_scheduler: Optional[nn.Module] = None,
    ):
        """Saves the model, optimizer, next epoch, and optionally the lr scheduler to a checkpoint file"""
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": current_epoch
            + 1,  # + 1 bc when we resume training we want to start at the next step
        }
        if lr_scheduler is not None:
            save_dict["lr_scheduler"] = lr_scheduler.state_dict()

        torch.save(
            save_dict,
            save_path,
        )


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification models"""

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def train_step(self, batch, criterion, grad_accum_steps):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = targets.to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.enable_amp,
        ):
            # (b, num_classes)
            preds = self.model(samples)

            loss = criterion(preds, targets)

            if grad_accum_steps > 1:
                # Scale the loss by the number of accumulation steps to average the gradients
                loss = loss / grad_accum_steps

        return samples, targets, preds, loss


def create_trainer(
    trainer_type: str,
    model: nn.Module,
    output_dir: str,
    step_lr_on: str,
    device: torch.device = torch.device("cpu"),
    log_train_steps: int = 20,
    amp_dtype: str = "float16",
    disable_amp: bool = False,
):
    """Initializes the trainer class based on the task type

    Args:
        trainer_type: the type of trainer to use; either "classification" or "ssl"
        see the Trainer subclass for more details on the specific arguments
    """
    if trainer_type == "classification":
        return ClassificationTrainer(
            model=model,
            output_dir=output_dir,
            step_lr_on=step_lr_on,
            device=device,
            log_train_steps=log_train_steps,
            amp_dtype=amp_dtype,
            disable_amp=disable_amp,
        )
    elif trainer_type == "simmim":
        return SimMIMTrainer()  # TODO: intialize
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

    ### start her build trainer class and test swin
