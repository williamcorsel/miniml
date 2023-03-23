from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import accuracy

from .models import *

class LeNetModule(LightningModule):
    def __init__(self, 
            num_classes=10,
            lr=0.001,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LeNet(num_classes=num_classes)
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.hparams.num_classes)

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
        optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer
    
    