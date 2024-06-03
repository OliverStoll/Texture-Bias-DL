import torch
from torch import nn
import wandb
import os
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics import F1Score, Precision, Recall
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
import logging

from utils.config import ROOT_DIR, CONFIG
from utils.logger import create_logger
from data_loading.dataloader import get_all_dataloader
from models.model_init import get_model
from callbacks.custom_progress_bar import CustomProgressBar
from _util_code import _set_gpus_from_config, _seed_everything, _mute_logs  # noqa


class ModelRunner(LightningModule):
    def __init__(self, dataset_name, dataset_config, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.task = dataset_config['task']
        self.dataset_name = dataset_name
        self.precision = Precision(num_classes=dataset_config['num_classes'],
                                   num_labels=dataset_config['num_labels'],
                                   average=CONFIG['score_average'],
                                   task=dataset_config['task'])
        self.recall = Recall(num_classes=dataset_config['num_classes'],
                             num_labels=dataset_config['num_labels'],
                             average=CONFIG['score_average'],
                             task=dataset_config['task'])
        self.f1_score = F1Score(num_classes=dataset_config['num_classes'],
                                num_labels=dataset_config['num_labels'],
                                average=CONFIG['score_average'],
                                task=dataset_config['task'])
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def calculate_metrics(self, batch):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        if self.task == 'multiclass':
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        elif self.task == 'multilabel':
            predictions = torch.sigmoid(logits)
        else:
            raise ValueError("Task not correctly passed through config")
        f1 = self.f1_score(predictions, labels.int())
        precision = self.precision(predictions, labels.int())
        recall = self.recall(predictions, labels.int())
        return loss, f1, precision, recall

    def training_step(self, batch, batch_idx):
        loss, f1, precision, recall = self.calculate_metrics(batch)
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('precision', precision, on_step=True, on_epoch=False, logger=True)
        self.log('recall', recall, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, f1, precision, recall = self.calculate_metrics(batch)
        self.log('val_loss', loss)
        self.log('val_f1', f1)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        if batch_idx == 0:
            self._log_image(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, f1, precision, recall = self.calculate_metrics(batch)
        self.log('test_loss', loss)
        self.log('test_f1', f1, prog_bar=True)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        if batch_idx == 0:
            self._log_image(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=CONFIG['learning_rate'])

    def _log_image(self, batch):
        images, labels = batch
        first_image = images[0]
        first_label = labels[0]

        first_prediction = self.forward(first_image.unsqueeze(0))
        predicted_label = torch.argmax(first_prediction, dim=1).item()

        if self.dataset_name == 'bigearthnet':
            first_image = first_image[[3, 2, 1], :, :]

        self.logger.experiment.log({"first_image_and_prediction": [
                wandb.Image(data_or_path=first_image,
                            caption=f"Prediction: {predicted_label}, Label: {first_label}")
            ]})

    def on_test_end(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass


def get_loss_fn(task):
    if task == 'multiclass':
        return nn.CrossEntropyLoss()
    elif task == 'multilabel':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Task not implemented / missing")


def init_everything_for_run(dataset_name, model_name, pretrained):
    dataset_config = CONFIG['datasets'][dataset_name]
    loss_fn = get_loss_fn(task=dataset_config['task'])
    model = get_model(model_name=model_name, dataset_config=dataset_config, pretrained=pretrained)
    model_runner = ModelRunner(model=model,
                               dataset_name=dataset_name,
                               dataset_config=dataset_config,
                               loss_fn=loss_fn)
    logger = WandbLogger(project=f"Master Thesis",
                         name=f"{model_name}-{dataset_name}{'-PT' if pretrained else ''}",
                         group=dataset_name,
                         tags=[f"model:{model_name}", f"data:{dataset_name}", 'is_pretrained' if pretrained else 'not_pretrained'],
                         save_dir=f'{ROOT_DIR}logs')
    profiler = SimpleProfiler(dirpath=f'{ROOT_DIR}logs/profiler',
                              filename=f'{dataset_name}-{model_name}',
                              extended=True)
    trainer = Trainer(max_epochs=CONFIG['epochs'],
                      limit_train_batches=CONFIG['limit_train_batches'],
                      limit_val_batches=CONFIG['limit_val_batches'],
                      limit_test_batches=CONFIG['limit_test_batches'],
                      logger=logger,
                      profiler=profiler,
                      enable_model_summary=False,
                      # callbacks=[CustomProgressBar()],
                      )

    return model_runner, trainer


def execute_run(dataset_name, model_name, dataloader_tuple, train=True, pretrained=False):
    log.info(f"STARTING RUN [ {dataset_name} | {model_name}{' | pretrained' if pretrained else ''} ]")
    train_loader, val_loader, test_loader = dataloader_tuple
    model_runner, trainer = init_everything_for_run(dataset_name, model_name, pretrained)
    if pretrained:
        log.info("PRETRAINED TEST")
        trainer.test(model_runner, test_loader)
    if train:
        log.info("TRAINING")
        trainer.fit(model_runner, train_loader, val_loader)
        trainer.test(model_runner, test_loader)
    log.info("FINISHED RUN\n\n\n")
    wandb.finish()



if __name__ == '__main__':
    log = create_logger("Main")
    models = ['vit', 'swin', 'convnext', 'resnet', 'efficientnet']
    datasets = ['imagenet', 'bigearthnet']

    dataloaders = get_all_dataloader(datasets)
    for dataset in datasets:
        for model in models:
            try:
                execute_run(dataset_name=dataset,
                            model_name=model,
                            dataloader_tuple=dataloaders[dataset],
                            train=True,
                            pretrained=(dataset == 'imagenet'))
            except Exception as e:
                log.error("Run failed", e)

