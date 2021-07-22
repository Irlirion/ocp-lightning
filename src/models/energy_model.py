from typing import Optional, Any, List
from cffi import model

import torch
from torch import nn

import pytorch_lightning as pl

from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from utils import Normalizer
from torch_geometric.data import Batch


class EnergyModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: int = 1e-3,
        weight_decay: float = 0.0,
        normalize_labels: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.L1Loss()
        self.normalize_labels = normalize_labels
        if self.normalize_labels:
            mean = 0 if mean is None else mean
            std = 1 if std is None else std
            self.normalizer = Normalizer(mean=mean, std=std, device=self.device)

        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x: Batch):
        output = self.model(x)

        if output.shape[-1] == 1:
            output = output.view(-1)

        return output

    def step(self, batch: Batch):
        preds = self.forward(batch)
        targets = batch.y_relaxed

        if self.normalize_labels:
            norm_targets = self.normalizer.norm(targets)
            denorm_preds = self.normalizer.denorm(preds)
            loss = self.criterion(preds, norm_targets)
            return loss, denorm_preds, targets

        loss = self.criterion(preds, targets)

        return loss, preds, targets

    def training_step(self, batch: Batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log_dict(
            self.train_metrics(preds, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(
            self.val_metrics(preds, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Batch, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer], []
