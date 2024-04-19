import torch
from torch import nn
from torchvision.models import resnet18
# from torchvision.models.resnet import ResNet18_Weights
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import F1Score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

from utils.config import ROOT_DIR, CONFIG
from data_loading.bigearthnet.BENv2DataModule import get_BENv2_dataloader
from _utils.gpus import set_gpus
from _utils.sanity_checks import print_dataloader_sizes


CONFIG_BEN = CONFIG['datasets']['bigearthnet']


class BigEarthNetResNet(LightningModule):
    def __init__(self, weights=None):
        super().__init__()
        base_model = resnet18(weights=weights)

        # Modify the first layer and last layer to fit the BigEarthNet dataset
        base_model.conv1 = nn.Conv2d(CONFIG_BEN['input_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, CONFIG_BEN['num_classes'])

        self.model = base_model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.f1_score = F1Score(num_labels=CONFIG_BEN['num_classes'],
                                average=CONFIG['f1_average'], threshold=0.5, task='multilabel')
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        preds = torch.sigmoid(logits)
        f1 = self.f1_score(preds, labels.int())
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels.float())
        preds = torch.sigmoid(logits)
        f1 = self.f1_score(preds, labels.int())
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=CONFIG['learning_rate'])





if __name__ == '__main__':
    set_gpus(CONFIG['gpu_indexes'])

    logger = WandbLogger(project="BigEarthNet-Resnet", save_dir=f'{ROOT_DIR}logs')
    model = BigEarthNetResNet()
    profiler = AdvancedProfiler()
    train_loader, val_loader, test_loader = get_BENv2_dataloader()
    print_dataloader_sizes(train_loader, val_loader, test_loader)

    trainer = Trainer(max_epochs=CONFIG['epochs'],
                      # limit_train_batches=500,
                      # limit_val_batches=200,
                      logger=logger,
                      profiler=profiler)
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=None)

