import itertools
import torch.nn as nn

import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score, AveragePrecision
from sklearn.metrics import classification_report
from utils.config import CONFIG

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



class TrainingModule(LightningModule):
    loss_fns = {
        'multiclass': nn.CrossEntropyLoss(),
        'multilabel': nn.BCEWithLogitsLoss()
    }

    def __init__(self, dataset_name, dataset_config, model):
        super().__init__()
        self.model = model
        self.task = dataset_config['task']
        self.loss_fn = self.loss_fns[self.task]
        self.dataset_name = dataset_name
        self.score_average = CONFIG['score_average']
        metric_data = {
            'num_classes': dataset_config['num_classes'],
            'num_labels': dataset_config['num_labels'],
            'task': dataset_config['task']
        }
        self.simple_metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.log_train_metrics = {'on_step': True, 'on_epoch': True, 'logger': True}
        self.log_val_metrics = {'on_step': False, 'on_epoch': True, 'logger': True}
        self.accuracy_micro = Accuracy(**metric_data, average='micro')
        self.precision_micro = Precision(**metric_data, average='micro')
        self.recall_micro = Recall(**metric_data, average='micro')
        self.f1_score_micro = F1Score(**metric_data, average='micro')
        self.map_metric_weighted = AveragePrecision(**metric_data, average='weighted')
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

    def forward(self, x):
        return self.model(x)

    def calculate_metrics(self, batch, is_eval=False):
        images, labels_float = batch
        unnormalized_logits = self(images)
        loss = self.loss_fn(unnormalized_logits, labels_float)
        labels = labels_float.int()
        if self.task == 'multiclass':
            predictions = torch.argmax(torch.softmax(unnormalized_logits, dim=1), dim=1)
            predictions_binary = predictions
        elif self.task == 'multilabel':
            predictions = torch.sigmoid(unnormalized_logits)
            predictions_binary = predictions.gt(0.5)
        else:
            raise ValueError("Task not correctly passed through config")
        metrics = {
            'loss': loss,
            'f1': self.f1_score_micro(predictions, labels),
            'precision': self.precision_micro(predictions, labels),
            'recall': self.recall_micro(predictions, labels),
            'accuracy': self.accuracy_micro(predictions, labels),
            'f1_macro': self.f1_score_macro(predictions, labels),
            'precision_macro': self.precision_macro(predictions, labels),
            'recall_macro': self.recall_macro(predictions, labels),
            'accuracy_macro': self.accuracy_macro(predictions, labels),
            'f1_classes': self.f1_score_classes(predictions, labels),
            'precision_classes': self.precision_classes(predictions, labels),
            'recall_classes': self.recall_classes(predictions, labels),
            'accuracy_classes': self.accuracy_classes(predictions, labels)
        }
        if is_eval:
            self.val_step_outputs.append((predictions_binary.cpu(), labels.cpu()))
            if self.task == 'multilabel':
                self.map_metric_weighted.update(predictions, labels)
                self.map_metric_macro.update(predictions, labels)
                self.map_metric_classes.update(predictions, labels)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch)
        self.log('loss', metrics['loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for metric in self.simple_metrics:
            self.log(metric, metrics[metric], **self.log_train_metrics)
            self.log(f"{metric}_macro", metrics[f"{metric}_macro"], **self.log_train_metrics)
            if self.task == 'multilabel':
                metric_classes = {f"{metric}_{BEN_LABELS[i][1]}": value for i, value in
                                  enumerate(metrics[f"{metric}_classes"])}
                self.log_dict(metric_classes, **self.log_train_metrics)
        if batch_idx == 0:
            self._log_image(batch, stage='train')  # UNTESTED
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, is_eval=True)
        self.log('val_loss', metrics['loss'], on_epoch=True, logger=True)
        for metric in self.simple_metrics:
            self.log(f'val_{metric}', metrics[metric], **self.log_val_metrics)
            self.log(f"val_{metric}_macro", metrics[f"{metric}_macro"], **self.log_val_metrics)
            if self.task == 'multilabel':
                metric_classes = {f"val_{metric}_{BEN_LABELS[i][1]}": value for i, value in
                                  enumerate(metrics[f"{metric}_classes"])}
                self.log_dict(metric_classes, **self.log_val_metrics)
        if batch_idx == 0:
            self._log_image(batch, stage='val')
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, is_eval=True)
        self.log('test_loss', metrics['loss'], logger=True, on_epoch=True)
        for metric in self.simple_metrics:
            self.log(f'test_{metric}', metrics[metric], **self.log_val_metrics)
            self.log(f"test_{metric}_macro", metrics[f"{metric}_macro"], **self.log_val_metrics)
            if self.task == 'multilabel':
                metric_classes = {f"test_{metric}_{BEN_LABELS[i][1]}": value for i, value in
                                  enumerate(metrics[f"{metric}_classes"])}
                self.log_dict(metric_classes, **self.log_val_metrics)
        if batch_idx == 0:
            self._log_image(batch, stage='test')
        return metrics['loss']

    def on_validation_epoch_end(self) -> None:
        self._log_classification_report()
        if self.task == 'multilabel':
            self.log('val_mAP', self.map_metric_weighted.compute(), logger=True, on_epoch=True)
            self.log('val_mAP_macro', self.map_metric_macro.compute(), logger=True, on_epoch=True)
            map_classes = self.map_metric_classes.compute()
            map_classes_dict = {f'val_mAP_{BEN_LABELS[i][1]}': map_class for i, map_class in enumerate(map_classes)}
            self.log_dict(map_classes_dict, on_epoch=True, logger=True)
            self.map_metric_weighted.reset()
            self.map_metric_macro.reset()
            self.map_metric_classes.reset()

    def on_test_epoch_end(self) -> None:
        self._log_classification_report()
        if self.task == 'multilabel':
            self.log('test_mAP', self.map_metric_weighted.compute(), logger=True, on_epoch=True)
            self.log('test_mAP_macro', self.map_metric_macro.compute(), logger=True, on_epoch=True)
            map_classes = self.map_metric_classes.compute()
            map_classes_dict = {f'test_mAP_{BEN_LABELS[i][1]}': map_class for i, map_class in enumerate(map_classes)}
            self.log_dict(map_classes_dict, on_epoch=True, logger=True)
            self.map_metric_weighted.reset()
            self.map_metric_macro.reset()
            self.map_metric_classes.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=CONFIG['learning_rate'])

    def _get_log_image_caption(self, image, labels, idx):
        raw_prediction = self.forward(image.unsqueeze(idx))
        if self.task == 'multilabel':
            label = [bool(label) for label in labels[idx].tolist()]
            prediction = torch.sigmoid(raw_prediction)
            prediction = prediction.gt(0.5).tolist()[0]
            pred_range = range(len(prediction))
            tp_pred = [BEN_LABELS[i][1] or BEN_LABELS[i][0] for i in pred_range if
                       prediction[i] and label[i]]
            fp_pred = [BEN_LABELS[i][1] or BEN_LABELS[i][0] for i in pred_range if
                       prediction[i] and not label[i]]
            fn_pred = [BEN_LABELS[i][1] or BEN_LABELS[i][0] for i in pred_range if
                       not prediction[i] and label[i]]
            caption = f"TP: {tp_pred}\n, FP: {fp_pred}\n, FN: {fn_pred}"
        elif self.task == 'multiclass':
            label = labels[idx].item()
            prediction = torch.softmax(raw_prediction, dim=1).tolist()
            caption = f"Prediction: {prediction}, Label: {label}"
        else:
            raise ValueError("Task not correctly passed through config")
        return caption

    def _log_image(self, batch, stage, idx=0):
        images, labels = batch
        image = images[idx]
        caption = self._get_log_image_caption(image, labels, idx)
        image = image if self.dataset_name == 'imagenet' else images[idx][[3, 2, 1], :, :]
        self.logger.experiment.log({f"{stage}_image": wandb.Image(data_or_path=image, caption=caption)})

    def _log_classification_report(self):
        predictions, labels = zip(*self.val_step_outputs)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        target_names = [label[1] for label in BEN_LABELS.values()] if self.task == 'multilabel' else None
        report = classification_report(labels.numpy(), predictions.numpy(), zero_division=0,
                                       output_dict=True, target_names=target_names)
        report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_table.add_data(label, metrics["precision"], metrics["recall"],
                                      metrics["f1-score"], metrics["support"])
        self.logger.experiment.log({f"classification_report": report_table})
        self.val_step_outputs.clear()