import torch
import pytorch_lightning as pl
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn
import os
import wandb
from pytorch_lightning.loggers import WandbLogger

from datasets import DataLoaderFactory
from ssl_models.mae import MAE


class MAEModel(pl.LightningModule):
    def __init__(self, vit, lr=1e-4):
        super().__init__()
        self.model = MAE(vit=vit)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, images):
        return self.model(images)

    def log_image(self, images, output_images, stage):
        image = images[0].cpu().clone()
        output_image = output_images[0].cpu().clone()
        output_image = torch.clamp(output_image, -5, 5)
        output_image = (output_image + 5) / 10
        self.logger.experiment.log({f'{stage}_input_image': wandb.Image(image)})
        self.logger.experiment.log({f'{stage}_output_image': wandb.Image(output_image)})

    def calculate_loss(self, batch, batch_idx, stage):
        images, _ = batch
        predictions, targets, output_images = self(images)
        loss = self.criterion(predictions, targets)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self.log_image(images, output_images, stage)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def _init_logger(logs_path='/home/olivers/colab-master-thesis/logs/ssl'):
    os.makedirs(logs_path, exist_ok=True)
    tags = ['ssl', 'data:imagenet', 'model:mae-vit']
    logger = WandbLogger(project=f"Master Thesis", save_dir=logs_path, tags=tags)
    return logger


if __name__ == '__main__':
    # Load a pre-trained ViT model
    devices = [2]
    epochs = 30
    logger = _init_logger()
    vit = vit_base_patch32_224(pretrained=True)
    mae_module = MAEModel(vit=vit)
    train_loader, val_loader, test_loader = DataLoaderFactory().get_dataloader('imagenet', 'vit')
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=devices,
        logger=logger,
        # limit_train_batches=5000,
        # limit_val_batches=250,
        # limit_test_batches=500
    )
    trainer.fit(mae_module, train_loader, val_loader)
    trainer.test(mae_module, test_loader)