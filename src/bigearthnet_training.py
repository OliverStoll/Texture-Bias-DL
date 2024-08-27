import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import F1Score, AveragePrecision
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks import StochasticWeightAveraging

from data_loading.BENv2DataModule import BENv2DataModule
from model_init import ModelCollection
from utils.config import CONFIG

wandb.require("core")


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


class GenericModule(LightningModule):
    model_collection = ModelCollection()
    INPUT_CHANNELS = 14
    OUTPUT_CLASSES = 19
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam

    def __init__(self, model_name, dataset_config, pretrained):
        super().__init__()
        self.model = self.model_collection.get_model(model_name, dataset_config, pretrained)
        # metrics
        metric_arguments = {'num_labels': self.OUTPUT_CLASSES, 'task': 'multilabel'}
        self.f1_micro = F1Score(average='micro', **metric_arguments)
        self.f1_macro = F1Score(average='macro', **metric_arguments)
        self.mAP_micro = AveragePrecision(average='micro', **metric_arguments)  # noqa
        self.mAP_macro = AveragePrecision(average='macro', **metric_arguments)

    def _calculate_metrics_on_step(self, batch, batch_idx, stage):
        images, labels_float = batch
        unnormalized_logits = self(images)
        labels = labels_float.int()
        predictions = torch.sigmoid(unnormalized_logits)

        self._log_image_if_first_batch(batch, batch_idx)
        if stage != "train":
            self.mAP_micro.update(predictions, labels)
            self.mAP_macro.update(predictions, labels)
        metrics = {
            f'{stage}_loss': self.loss_fn(unnormalized_logits, labels_float),
            f'{stage}_f1_micro': self.f1_micro(predictions, labels),
            f'{stage}_f1_macro': self.f1_macro(predictions, labels),
        }
        return metrics

    def _log_mAP_scores_on_epoch_end(self, stage):
        try:
            self.log(f'{stage}_mAP_micro', self.mAP_micro.compute(), on_step=False, on_epoch=True, logger=True)
            self.log(f'{stage}_mAP_macro', self.mAP_macro.compute(), on_step=False, on_epoch=True, logger=True)
        except Exception as e:
            print("mAP nicht updated?!", e)
        self.mAP_macro.reset()
        self.mAP_micro.reset()

    def _log_image_if_first_batch(self, batch, batch_idx, image_idx=0):
        if batch_idx != 0:
            return
        images, labels = batch
        image = images[image_idx]
        raw_prediction = self(image.unsqueeze(0))
        prediction = torch.sigmoid(raw_prediction)
        pred_label_zip = list(zip(prediction[0], labels[image_idx]))
        prediction_text = ', '.join([f"{'#' if l else ''}{int(p*10)}" for p, l in pred_label_zip])
        caption = f"Preds:  {prediction_text}"
        image = image[[3, 2, 1], :, :]
        self.logger.experiment.log({'image': wandb.Image(data_or_path=image, caption=caption)})  # noqa

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        metrics = self._calculate_metrics_on_step(batch, batch_idx, stage="train")
        self.log_dict(metrics, logger=True, prog_bar=True)
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._calculate_metrics_on_step(batch, batch_idx, stage="val")
        self.log_dict(metrics, logger=True)
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        metrics = self._calculate_metrics_on_step(batch, batch_idx, stage="test")
        self.log_dict(metrics, logger=True)
        return metrics['test_loss']

    def on_validation_epoch_end(self):
        self._log_mAP_scores_on_epoch_end("val")

    def on_test_epoch_end(self):
        self._log_mAP_scores_on_epoch_end("test")


def train_model_on_bigearthnet(model_name, dl_tuple):
    train_dl, val_dl, test_dl = dl_tuple
    pl.seed_everything(42)
    bigearthnet_config = CONFIG['datasets']['bigearthnet']
    tags = [f"model:{model_name}", f"dataset:bigearthnet", f"img_size:{IMG_SIZE}", f"batch_size:{BATCH_SIZE}", f"lr:{LR}", f"num_workers:{NUM_WORKERS}", f"gradient_clipping:{GRADIENT_CLIP_VAL}"]

    module = GenericModule(model_name=model_name, dataset_config=bigearthnet_config, pretrained=False)
    wandb_logger = WandbLogger(name=model_name, project='bigearthnet', tags=tags, save_dir=LOGGING_PATH)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mAP_micro',
        dirpath=f'{LOGGING_PATH}/checkpoints/{model_name}',
        filename='{epoch:02d}-{val_mAP_micro:.3f}',
        mode='max'
    )

    trainer = Trainer(max_epochs=EPOCHS,
                      logger=wandb_logger,
                      devices=DEVICES,
                      gradient_clip_val=GRADIENT_CLIP_VAL,
                      callbacks=[checkpoint_callback],  # [SWA]
                      fast_dev_run=False,
                      limit_train_batches=None,
                      limit_val_batches=None,
                      )
    trainer.fit(module, train_dl, val_dl)
    trainer.test(module, test_dl)
    wandb.finish()



if __name__ == '__main__':
    LOGGING_PATH = '/media/storagecube/olivers/logs'
    # train variables
    EPOCHS = 10
    DEVICES = [3]
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_WORKERS = 30
    IMG_SIZE = 112

    # improvements
    # GRADIENT_CLIP_VAL = 0.5

    # train the model
    print(torch.version.cuda)
    _dl_tuple = get_bigearthnet_dataloader()
    for _model_name in ['resnet', 'convnext', 'efficientnet', 'vit', 'swin']:
        for GRADIENT_CLIP_VAL in [None, 0.5]:
            print(f"Training model {_model_name}")
            try:
                train_model_on_bigearthnet(model_name=_model_name, dl_tuple=_dl_tuple)
            except Exception as e:
                print(f"Error in training model {_model_name}: {e}")