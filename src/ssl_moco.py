import torch
import torch.nn as nn
from lightly.models import MoCo
from lightly.loss import NTXentLoss
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet50

from datasets import DataLoaderFactory


class MoCoModel(pl.LightningModule):
    def __init__(self, backbone, num_ftrs, out_dim, momentum=0.999, temperature=0.2):
        super().__init__()
        self.backbone = backbone
        self.criterion = NTXentLoss(temperature=temperature)
        self.moco = MoCo(
            backbone=self.backbone,
            num_ftrs=num_ftrs,
            out_dim=out_dim,
            m=momentum,
            batch_shuffle=True
        )

    def forward(self, x):
        return self.moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0, z1, _ = self.moco(x0, x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        return optimizer


if __name__ == '__main__':
    # TODO: create a PairedDataset class for the dataloading
    # load backbone resnet with pretrained weights
    DEVICES = [2]
    train_dl, val_dl, _ = DataLoaderFactory().get_dataloader(dataset_name='imagenet', model_name='resnet')
    resnet_model = resnet50(weights='DEFAULT')
    num_features = resnet_model.fc.in_features
    out_dimension = 128
    model = MoCoModel(backbone=resnet_model, num_ftrs=num_features, out_dim=out_dimension)
    trainer = pl.Trainer(max_epochs=5, devices=DEVICES)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)