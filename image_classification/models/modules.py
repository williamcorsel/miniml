import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from . import *


class LeNetModule(LightningModule):
    def __init__(self, 
            num_classes=10,
            lr=0.05,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LeNet(num_classes=num_classes)
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task="multiclass", num_classes=self.hparams.num_classes)

        if stage is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), 
            lr=self.hparams.lr, 
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                steps_per_epoch=len(self.train_dataloader()),
                epochs=self.trainer.max_epochs,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def create_model():
    import torch.nn as nn
    import torchvision
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNetModule(LightningModule):
    def __init__(self, 
            num_classes=10,
            lr=0.001,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(num_classes=num_classes)
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task="multiclass", num_classes=self.hparams.num_classes)

        if stage is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")

    def predict_step(self, batch, batch_idx):
        imgs, _ = batch
        logits = self.model(imgs)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer
    