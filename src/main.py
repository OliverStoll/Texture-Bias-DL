import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics import F1Score, Precision, Recall
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from utils.config import ROOT_DIR, CONFIG
from _utils import set_gpus_from_config  # noqa
from data_loading.dataloader import get_dataloader
from models.model_init import get_model


class ModelRunner(LightningModule):
    def __init__(self, dataset_config, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.task = dataset_config['task']
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
        assert labels.min() >= 0
        assert labels.max() <= 999

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, f1, precision, recall = self.calculate_metrics(batch)
        self.log('val_loss', loss)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        return loss

    def test_step(self, batch, batch_idx):
        loss, f1, precision, recall = self.calculate_metrics(batch)
        self.log('test_loss', loss)
        self.log('test_f1', f1, prog_bar=True)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=CONFIG['learning_rate'])


def get_loss_fn(task):
    if task == 'multiclass':
        return nn.CrossEntropyLoss()
    elif task == 'multilabel':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Task not implemented / missing")


def get_everything_for_run(dataset_name, model_name, pretrained):
    seed_everything(seed=42)
    dataset_config = CONFIG['datasets'][dataset_name]
    loss_fn = get_loss_fn(task=dataset_config['task'])
    model = get_model(model_name=model_name, dataset_config=dataset_config, pretrained=pretrained)
    model_runner = ModelRunner(model=model,
                               dataset_config=dataset_config,
                               loss_fn=loss_fn)
    train_loader, val_loader, test_loader = get_dataloader(dataset_name=dataset_name)

    logger = WandbLogger(project=f"{dataset_name}-{model_name}", save_dir=f'{ROOT_DIR}logs')
    profiler = SimpleProfiler(dirpath=f'{ROOT_DIR}logs/profiler',
                              filename=f'{dataset_name}-{model_name}',
                              extended=True)
    trainer = Trainer(max_epochs=CONFIG['epochs'],
                      limit_train_batches=None,
                      limit_val_batches=CONFIG['limit_val_batches'],
                      logger=logger,
                      profiler=profiler)
    return model_runner, trainer, train_loader, val_loader, test_loader


def execute_run(dataset_name, model_name, pretrained, train=True):
    model_runner, trainer, train_loader, val_loader, test_loader = get_everything_for_run(dataset_name, model_name, pretrained)
    trainer.test(model_runner, test_loader)
    if train:
        trainer.fit(model_runner, train_loader, val_loader)
        trainer.test(model_runner, test_loader)



if __name__ == '__main__':
    execute_run(dataset_name='imagenet', model_name='resnet', pretrained=False, train=True)
