import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import F1Score, AveragePrecision

from data_loading.BENv2DataModule import BENv2DataModule


def get_bigearthnet_dataloader():
    """ Get the dataloaders, using https://git.tu-berlin.de/rsim/BENv2-DataLoading """
    files = {"train": "all_train.csv", "validation": "all_val.csv", "test": "all_test.csv"}
    keys = {}
    for split, file in files.items():
        keys[split] = pd.read_csv(
            f"/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits/{file}",
            header=None,
            names=["name"]).name.to_list()

    data_module = BENv2DataModule(
        keys=keys,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,  # the image-size is passed, and will be used as a final resize
        image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
        label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
        s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
        train_transforms=None,
        eval_transforms=None,
        interpolation_mode=None
    )
    data_module.setup()
    return data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()


class SwinModule(LightningModule):
    def __init__(self):
        super().__init__()
        # default arguments for Swin Transformer on BigEarthNet
        self.INPUT_CHANNELS = 14
        self.OUTPUT_CLASSES = 19
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize the Swin Transformer (and adapt it to the image channels and output classes)
        self.swin_model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swin_model.features[0][0] = nn.Conv2d(in_channels=self.INPUT_CHANNELS,
                                                   out_channels=96,
                                                   kernel_size=4,
                                                   stride=4)
        self.swin_model.head = nn.Linear(in_features=self.swin_model.head.in_features,
                                         out_features=self.OUTPUT_CLASSES)

        # Metrics
        self.f1_micro = F1Score(num_labels=self.OUTPUT_CLASSES, average='micro', task='multilabel')
        self.f1_macro = F1Score(num_labels=self.OUTPUT_CLASSES, average='macro', task='multilabel')
        self.mAP_weighted = AveragePrecision(num_labels=self.OUTPUT_CLASSES, average='weighted', task='multilabel')
        self.mAP_macro = AveragePrecision(num_labels=self.OUTPUT_CLASSES, average='macro', task='multilabel')

    def forward(self, x):
        return self.swin_model(x)

    def calculate_metrics(self, batch, run_type):
        images, labels_float = batch
        unnormalized_logits = self(images)
        loss = self.loss_fn(unnormalized_logits, labels_float)
        labels = labels_float.int()
        predictions = torch.sigmoid(unnormalized_logits)

        self.mAP_macro.update(predictions, labels)
        self.mAP_weighted.update(predictions, labels)
        metrics = {
            f'{run_type}_loss': loss,
            f'{run_type}_f1_micro': self.f1_micro(predictions, labels),
            f'{run_type}_f1_macro': self.f1_macro(predictions, labels),
        }
        return metrics

    def log_mAP_scores_on_epoch_end(self, run_type):
        self.log(f'{run_type}_mAP_macro', self.mAP_macro.compute(), on_epoch=True, logger=True)
        self.log(f'{run_type}_mAP_weighted', self.mAP_weighted.compute(), on_epoch=True, logger=True)
        self.mAP_macro.reset()
        self.mAP_weighted.reset()

    def log_image(self, batch, image_idx=0):
        images, labels = batch
        image = images[image_idx]
        raw_prediction = self(image.unsqueeze(0))
        prediction = torch.sigmoid(raw_prediction)
        image = image[[3, 2, 1], :, :]
        self.logger.experiment.log({'image': wandb.Image(image, caption=str(prediction.tolist()))})

    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, run_type="train")
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        if batch_idx == 0:
            self.log_image(batch)
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, run_type="val")
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, run_type="test")
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return metrics['test_loss']

    def on_train_epoch_end(self):
        self.log_mAP_scores_on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.log_mAP_scores_on_epoch_end("val")

    def on_test_epoch_end(self):
        self.log_mAP_scores_on_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


def train_swin_on_bigearthnet():
    train_dl, val_dl, test_dl = get_bigearthnet_dataloader()
    swin_module = SwinModule()
    wandb_logger = WandbLogger(project='swin_example')
    trainer = Trainer(max_epochs=EPOCHS,
                      logger=wandb_logger,
                      devices=DEVICES)
    trainer.fit(swin_module, train_dl, val_dl)



if __name__ == '__main__':
    pl.seed_everything(42)

    # train details
    EPOCHS = 20
    DEVICES = [3]
    BATCH_SIZE = 32
    LR = 1e-3
    NUM_WORKERS = 30
    IMG_SIZE = 224

    # train the model
    train_swin_on_bigearthnet()

    """
    Data transforms (applied in BENv2DataModule):
    
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean, std),  # mean and std are calculated from the dataset by the BENv2DataModule
    transforms.Resize((img_size, img_size)),  # passed as 224
    """