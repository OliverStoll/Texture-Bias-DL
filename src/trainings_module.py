import os

import torch.nn as nn
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score, AveragePrecision
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from common_utils.config import CONFIG
from common_utils.logger import create_logger

from data_loading.caltech.caltech_constants import CALTECH_CLASSNAMES_20

BEN_LABELS = {
    0: ("Urban fabric", "Urban"),
    1: ("Industrial or commercial units", "Industrial Units"),
    2: ("Arable land", "Arable"),
    3: ("Permanent crops", "Crops"),
    4: ("Pastures", "Pastures"),
    5: ("Complex cultivation patterns", "Cultivation"),
    6: ("Land principally occupied by agriculture, with significant areas of natural vegetation", "Agriculture"),
    7: ("Agro-forestry areas", "Agro-forestry"),
    8: ("Broad-leaved forest", "Broad-leaved"),
    9: ("Coniferous forest", "Coniferous"),
    10: ("Mixed forest", "Mixed forest"),
    11: ("Natural grassland and sparsely vegetated areas", "Grassland"),
    12: ("Moors, heathland and sclerophyllous vegetation", "Moors"),
    13: ("Transitional woodland, shrub", "Transitional woodland"),
    14: ("Beaches, dunes, sands", "Sand"),
    15: ("Inland wetlands", "Wetlands"),
    16: ("Coastal wetlands", "Coastal"),
    17: ("Inland waters", "Waters"),
    18: ("Marine waters", "Marine"),
}

DEEPGLOBE_LABELS = {
    0: "urban_land",
    1: "agricultural_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
}




class TrainingModule(LightningModule):
    log_ = create_logger("Training Module")
    loss_fns = {'multiclass': nn.CrossEntropyLoss(), 'multilabel': nn.BCEWithLogitsLoss()}
    optimizer = torch.optim.Adam
    log_mAP_classes = True  # DISABLED DUE TO CLUTTER
    all_target_names = {
        'bigearthnet': [label[1] for label in BEN_LABELS.values()],
        'rgb_bigearthnet': [label[1] for label in BEN_LABELS.values()],
        'deepglobe': [label for label in DEEPGLOBE_LABELS.values()],
        'caltech': [label for label in CALTECH_CLASSNAMES_20],
        'caltech_120': [label for label in CALTECH_CLASSNAMES_20],
        'caltech_ft': [label for label in CALTECH_CLASSNAMES_20],
        'imagenet': range(1000)
    }

    def __init__(self, dataset_name, dataset_config, model):
        super().__init__()
        self.model = model
        self.task = dataset_config['task']
        self.loss_fn = self.loss_fns[self.task]
        self.dataset_name = dataset_name
        self.score_average = CONFIG['score_average']
        # use get
        metric_data = {
            'num_classes': dataset_config.get('num_classes', None),
            'num_labels': dataset_config.get('num_labels', None),
            'task': dataset_config['task']
        }
        self.target_names = self.all_target_names[dataset_name]
        self.simple_metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.accuracy_micro = Accuracy(**metric_data, average='micro')
        self.precision_micro = Precision(**metric_data, average='micro')
        self.recall_micro = Recall(**metric_data, average='micro')
        self.f1_score_micro = F1Score(**metric_data, average='micro')
        map_micro_avg = 'micro' if dataset_name in ['bigearthnet', 'rgb_bigearthnet', 'deepglobe'] else 'weighted'  # TODO: fix?
        self.map_metric_micro = AveragePrecision(**metric_data, average=map_micro_avg)  # noqa
        # macro metrics
        self.accuracy_macro = Accuracy(**metric_data, average='macro')
        self.precision_macro = Precision(**metric_data, average='macro')
        self.recall_macro = Recall(**metric_data, average='macro')
        self.f1_score_macro = F1Score(**metric_data, average='macro')
        self.map_metric_macro = AveragePrecision(**metric_data, average='macro')
        # now use un-averaged scores
        self.accuracy_classes = Accuracy(**metric_data, average=None)
        self.precision_classes = Precision(**metric_data, average=None)
        self.recall_classes = Recall(**metric_data, average=None)
        self.f1_score_classes = F1Score(**metric_data, average=None)
        self.map_metric_classes = AveragePrecision(**metric_data, average=None)
        # save validation predictions for classification report
        self.val_step_outputs = []
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def _get_prediction(self, unnormalized_logits):
        if self.task == 'multiclass':
            predictions = torch.argmax(torch.softmax(unnormalized_logits, dim=1), dim=1)
            predictions_binary = predictions
        elif self.task == 'multilabel':
            predictions = torch.sigmoid(unnormalized_logits)
            predictions_binary = predictions.gt(0.5)
        else:
            raise ValueError("Task not correctly passed through dataset config")
        return predictions, predictions_binary

    def _calculate_map_metrics(self, predictions, predictions_binary, labels):
        self.val_step_outputs.append((predictions_binary.cpu(), labels.cpu()))
        if self.task == 'multilabel':
            self.map_metric_micro.update(predictions, labels)
            self.map_metric_macro.update(predictions, labels)
            self.map_metric_classes.update(predictions, labels)

    def _warn_if_images_unnormalized(self, images):
        if images.max() > 10 or images.min() < -10:
            self.log_.error(f"Images are not normalized: max: {images.max()}, min: {images.min()}")

    def calculate_metrics(self, batch, batch_idx, stage):
        images, labels_float = batch

        labels = labels_float.int()
        unnormalized_logits = self(images)
        loss = self.loss_fn(unnormalized_logits, labels_float)
        predictions, predictions_binary = self._get_prediction(unnormalized_logits)
        if stage == 'val' or stage == 'test':
            self._calculate_map_metrics(predictions, predictions_binary, labels)
        if batch_idx == 0:
            self.log_image(batch, stage=stage)

        metrics = {
            f'{stage}_loss': loss,
            # f'{stage}_f1_micro': self.f1_score_micro(predictions, labels),
            # f'{stage}_precision_micro': self.precision_micro(predictions, labels),
            # f'{stage}_recall_micro': self.recall_micro(predictions, labels),
            f'{stage}_accuracy_micro': self.accuracy_micro(predictions, labels),
            # f'{stage}_f1_macro': self.f1_score_macro(predictions, labels),
            # f'{stage}_precision_macro': self.precision_macro(predictions, labels),
            # f'{stage}_recall_macro': self.recall_macro(predictions, labels),
            f'{stage}_accuracy_macro': self.accuracy_macro(predictions, labels),
            # f'{stage}_f1_classes': self.f1_score_classes(predictions, labels),
            # f'{stage}_precision_classes': self.precision_classes(predictions, labels),
            # f'{stage}_recall_classes': self.recall_classes(predictions, labels),
            f'{stage}_accuracy_classes': self.accuracy_classes(predictions, labels)
        }
        return metrics

    def log_mAP_on_epoch_end(self, stage):
        if self.task == 'multilabel':
            self.log(f'{stage}_mAP_micro', self.map_metric_micro.compute(), logger=True, on_epoch=True)
            self.log(f'{stage}_mAP_macro', self.map_metric_macro.compute(), logger=True, on_epoch=True)
            map_classes = self.map_metric_classes.compute()
            map_classes_dict = {f'{stage}_mAP_{BEN_LABELS[i][1]}': map_class for i, map_class in enumerate(map_classes)}
            if self.log_mAP_classes:
                self.log_dict(map_classes_dict, on_epoch=True, logger=True)
            self.map_metric_micro.reset()
            self.map_metric_macro.reset()
            self.map_metric_classes.reset()

    def _get_log_image_caption(self, image, labels, idx):
        raw_prediction = self.forward(image.unsqueeze(idx))
        if self.task == 'multilabel':
            label = [bool(label) for label in labels[idx].tolist()]
            prediction = torch.sigmoid(raw_prediction)
            prediction = prediction.gt(0.5).tolist()[0]
            pred_range = range(len(prediction))
            tp_pred = [self.target_names[i] for i in pred_range if prediction[i] and label[i]]
            fp_pred = [self.target_names[i] for i in pred_range if prediction[i] and (not label[i])]
            fn_pred = [self.target_names[i] for i in pred_range if (not prediction[i]) and label[i]]
            caption = f"TP: {tp_pred}\n, FP: {fp_pred}\n, FN: {fn_pred}"
        elif self.task == 'multiclass':
            label = labels[idx].item()
            prediction = torch.softmax(raw_prediction, dim=1).tolist()
            prediction = [round(p, 1) for p in prediction[0]]
            caption = f"Prediction: {prediction}, Label: {label}"
        else:
            raise ValueError("Task not correctly passed through config")
        return caption

    def log_image(self, batch, stage, idx=0):
        if os.environ.get('DISABLE_ARTIFACTS', False):
            return
        images, labels = batch
        image = images[idx]
        caption = self._get_log_image_caption(image, labels, idx)
        if image.shape[0] > 3 and image.shape[0] < 100:
            image = image[[3, 2, 1], :, :]
        self.logger.experiment.log({f"{stage}_image": wandb.Image(data_or_path=image, caption=caption)})

    def log_classification_report(self, stage):
        try:
            predictions, labels = zip(*self.val_step_outputs)
            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            report = classification_report(labels.numpy(), predictions.numpy(), zero_division=0,
                                           output_dict=True, target_names=self.target_names)
            report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    report_table.add_data(label, metrics["precision"], metrics["recall"],
                                          metrics["f1-score"], metrics["support"])
            self.logger.experiment.log({f"{stage}_classification_report": report_table})
        except Exception as e:
            self.log_.warning("Error logging classification report")
        finally:
            self.val_step_outputs.clear()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='train')
        train_loss = metrics.pop('train_loss')
        self.log('train_loss', train_loss, prog_bar=True)
        self.log_dict(metrics, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='val')
        self.log_dict(metrics, logger=True)
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='test')
        self.log_dict(metrics, logger=True)
        return metrics['test_loss']

    def on_validation_epoch_end(self) -> None:
        self.log_mAP_on_epoch_end('val')
        self.log_classification_report('val')

    def on_test_epoch_end(self) -> None:
        self.log_mAP_on_epoch_end('test')
        self.log_classification_report('test')

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=CONFIG['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return [optimizer], [scheduler]