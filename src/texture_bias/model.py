import pytorch_lightning as pl
import torchmetrics
from torch import nn
from torchvision import models


class PretrainedResnet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, num_classes)
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        return self.resnet_model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.f1_score(scores, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            {"test_f1": self.f1_score.compute()}, on_epoch=True, prog_bar=True, logger=True
        )
