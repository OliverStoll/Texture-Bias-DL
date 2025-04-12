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

from data_loading.caltech.constants import CALTECH_CLASSNAMES_20
from data_loading.deepglobe.constants import DEEPGLOBE_LABELS
from data_loading.bigearthnet.constants import BEN_LABELS


class TrainingModule(LightningModule):
    """
    PyTorch Lightning module for classification tasks across various datasets.

    Supports both multiclass and multilabel classification, and automatically selects
    the appropriate loss function and label mappings based on dataset configuration.
    """

    log_ = create_logger("Training Module")
    optimizer = torch.optim.Adam
    lr_scheduler_kwargs = {
        'T_max': 50,
        'eta_min': 1e-5
    }
    loss_functions = {
        'multiclass': nn.CrossEntropyLoss(),
        'multilabel': nn.BCEWithLogitsLoss()
    }
    def __init__(
        self,
        dataset_name: str,
        dataset_config: dict,
        model: nn.Module,
        learning_rate: float = CONFIG['learning_rate'],
        log_map_classes: bool = True,
        score_average: str = CONFIG['score_average'],
    ):
        """
        Initialize the training module.

        Args:
            dataset_name: Identifier of the dataset in use.
            dataset_config: Configuration dictionary for the dataset.
            model: The model to train.
            log_map_classes: Whether to log class-wise mAP during validation.
        """
        super().__init__()
        self.model = model
        self.dataset_name = dataset_name
        self.task = dataset_config['task']
        self.loss_fn = self.loss_functions[self.task]
        self.learning_rate = learning_rate
        self.log_mAP_classes = log_map_classes
        self.score_average = score_average
        self.val_step_outputs = []

        self._initialize_metrics(dataset_name, dataset_config)
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    datasets_label_names = {
        'bigearthnet': [entry["short"] for entry in BEN_LABELS.values()],
        'rgb_bigearthnet': [entry["short"] for entry in BEN_LABELS.values()],
        'deepglobe': list(DEEPGLOBE_LABELS.values()),
        'caltech': list(CALTECH_CLASSNAMES_20),
        'caltech_120': list(CALTECH_CLASSNAMES_20),
        'caltech_ft': list(CALTECH_CLASSNAMES_20),
        'imagenet': list(range(1000))
    }

    def _initialize_metrics(self, dataset_name: str, dataset_config: dict) -> None:
        """
        Initialize classification metrics for micro, macro, and per-class evaluation.

        Args:
            dataset_name: Name of the dataset to determine label mappings and averaging strategy.
            dataset_config: Dictionary specifying task type, number of classes/labels, etc.
        """
        metric_base = {
            'num_classes': dataset_config.get('num_classes'),
            'num_labels': dataset_config.get('num_labels'),
            'task': dataset_config['task']
        }
        metric_micro = {**metric_base, 'average': 'micro'}
        metric_macro = {**metric_base, 'average': 'macro'}

        self.target_names = self.datasets_label_names[dataset_name]

        # Micro-averaged metrics
        self.accuracy_micro = Accuracy(**metric_micro)
        self.precision_micro = Precision(**metric_micro)
        self.recall_micro = Recall(**metric_micro)
        self.f1_score_micro = F1Score(**metric_micro)
        self.map_metric_micro = AveragePrecision(**metric_base, average='weighted')

        # Macro-averaged metrics
        self.accuracy_macro = Accuracy(**metric_macro)
        self.precision_macro = Precision(**metric_macro)
        self.recall_macro = Recall(**metric_macro)
        self.f1_score_macro = F1Score(**metric_macro)
        self.map_metric_macro = AveragePrecision(**metric_macro)

        # Per-class (unaveraged) metrics
        self.accuracy_classes = Accuracy(**metric_base)
        self.precision_classes = Precision(**metric_base)
        self.recall_classes = Recall(**metric_base)
        self.f1_score_classes = F1Score(**metric_base)
        self.map_metric_classes = AveragePrecision(**metric_base)

    def _get_prediction(self, unnormalized_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw model outputs (logits) into probability predictions and binary predictions.

        Returns:
            A tuple of (probabilistic predictions, binary class predictions).
        """
        if self.task == 'multiclass':
            probabilities = torch.softmax(unnormalized_logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            binary_predictions = predicted_classes

        elif self.task == 'multilabel':
            probabilities = torch.sigmoid(unnormalized_logits)
            binary_predictions = probabilities.gt(0.5)

        else:
            raise ValueError("Task not correctly passed through dataset config.")

        return probabilities, binary_predictions

    def _update_map_metrics(
        self,
        predictions: torch.Tensor,
        predictions_binary: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """
        Update Average Precision (mAP) metrics based on predictions and ground truth labels.

        Stores binary predictions and labels for later use (e.g., mAP computation across epoch).

        Args:
            predictions: Probability/logit outputs from the model.
            predictions_binary: Binarized predictions (e.g., thresholded at 0.5 for multilabel).
            labels: Ground truth labels.
        """
        self.val_step_outputs.append((predictions_binary.cpu(), labels.cpu()))

        if self.task == 'multilabel':
            self.map_metric_micro.update(predictions, labels)
            self.map_metric_macro.update(predictions, labels)
            self.map_metric_classes.update(predictions, labels)

    def _warn_if_images_unnormalized(self, images: torch.Tensor) -> None:
        """
        Log an error if image tensors appear to be unnormalized.

        Checks whether any pixel value is outside the expected normalized range.

        Args:
            images: A batch of image tensors.
        """
        if images.max() > 10 or images.min() < -10:
            self.log_.error(
                f"Images are not normalized: max = {images.max().item():.2f}, min = {images.min().item():.2f}"
            )

    def calculate_metrics(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            stage: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute metrics and loss for a given batch and training stage.

        Args:
            batch: A tuple of (images, labels) from the dataloader.
            batch_idx: Index of the current batch.
            stage: One of 'train', 'val', or 'test', used to prefix logged metrics.

        Returns:
            A dictionary of computed metrics and the class-wise accuracy score.
        """
        images, labels_float = batch
        labels = labels_float.int()

        self._warn_if_images_unnormalized(images)

        unnormalized_logits = self(images)
        loss = self.loss_fn(unnormalized_logits, labels_float)
        predictions, predictions_binary = self._get_prediction(unnormalized_logits)

        if stage in {"val", "test"}:
            self._update_map_metrics(predictions, predictions_binary, labels)

        if batch_idx == 0:
            self.log_image(batch, stage=stage)

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_f1_micro": self.f1_score_micro(predictions, labels),
            f"{stage}_precision_micro": self.precision_micro(predictions, labels),
            f"{stage}_recall_micro": self.recall_micro(predictions, labels),
            f"{stage}_accuracy_micro": self.accuracy_micro(predictions, labels),
            f"{stage}_f1_macro": self.f1_score_macro(predictions, labels),
            f"{stage}_precision_macro": self.precision_macro(predictions, labels),
            f"{stage}_recall_macro": self.recall_macro(predictions, labels),
            f"{stage}_accuracy_macro": self.accuracy_macro(predictions, labels),
            f"{stage}_f1_classes": self.f1_score_classes(predictions, labels),
            f"{stage}_precision_classes": self.precision_classes(predictions, labels),
            f"{stage}_recall_classes": self.recall_classes(predictions, labels),
            f"{stage}_accuracy_classes": self.accuracy_classes(predictions, labels),
        }

        return metrics, self.accuracy_classes(predictions, labels)

    def log_map_on_epoch_end(self, stage: str) -> None:
        """
        Log mAP (mean Average Precision) metrics at the end of an epoch for multilabel tasks.

        Args:
            stage: One of 'train', 'val', or 'test', used to prefix logged metrics.
        """
        if self.task != 'multilabel':
            return
        # Log micro and macro averaged mAP
        self.log(f'{stage}_mAP_micro', self.map_metric_micro.compute(), logger=True, on_epoch=True)
        self.log(f'{stage}_mAP_macro', self.map_metric_macro.compute(), logger=True, on_epoch=True)
        # Log class-wise mAP
        map_classes = self.map_metric_classes.compute()
        map_classes_dict = {
            f'{stage}_mAP_{BEN_LABELS[i]["short"]}': score
            for i, score in enumerate(map_classes)
        }
        if self.log_mAP_classes:
            self.log_dict(map_classes_dict, on_epoch=True, logger=True)
        # Reset all metrics
        self.map_metric_micro.reset()
        self.map_metric_macro.reset()
        self.map_metric_classes.reset()


    def _get_log_image_caption(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        idx: int
    ) -> str:
        """
        Generate a caption summarizing predictions vs. ground truth for a single image.

        Args:
            image: A single image tensor from the batch.
            labels: The batch of ground truth labels.
            idx: Index of the image in the batch.

        Returns:
            A string caption describing TP/FP/FN (multilabel) or class probabilities (multiclass).
        """
        raw_prediction = self.forward(image.unsqueeze(idx))
        if self.task == 'multilabel':
            label = [bool(label) for label in labels[idx].tolist()]
            prediction = torch.sigmoid(raw_prediction)
            prediction = prediction.gt(0.5).tolist()[0]
            prediction_range = range(len(prediction))
            tp_pred = [self.target_names[i] for i in prediction_range if prediction[i] and label[i]]
            fp_pred = [self.target_names[i] for i in prediction_range if prediction[i] and (not label[i])]
            fn_pred = [self.target_names[i] for i in prediction_range if (not prediction[i]) and label[i]]
            caption = f"TP: {tp_pred}\n, FP: {fp_pred}\n, FN: {fn_pred}"
        elif self.task == 'multiclass':
            label = labels[idx].item()
            prediction = torch.softmax(raw_prediction, dim=1).tolist()
            prediction = [round(p, 1) for p in prediction[0]]
            caption = f"Prediction: {prediction}, Label: {label}"
        else:
            raise ValueError("Task not correctly passed through config")

        return caption

    def log_image(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
        batch_idx: int = 0
    ) -> None:
        """
        Log a single image with caption to the experiment logger (e.g., Weights & Biases).

        Skips logging if the `DISABLE_ARTIFACTS` environment variable is set.

        Args:
            batch: A tuple of (images, labels) from the current step.
            stage: The current phase ('train', 'val', or 'test') for naming the log entry.
            batch_idx: Index of the image within the batch to visualize.
        """
        if os.environ.get('DISABLE_ARTIFACTS', False):
            return
        images, labels = batch
        image = images[batch_idx]
        if image.shape[0] > 3:
            image = image[[3, 2, 1], :, :]
        caption = self._get_log_image_caption(image, labels, batch_idx)
        self.logger.experiment.log(
            {f"{stage}_image": wandb.Image(data_or_path=image, caption=caption)}
        )

    def log_classification_report(self, stage: str) -> None:
        """
        Log a per-class classification report as a W&B table.

        Uses collected predictions and labels from `val_step_outputs` to compute
        precision, recall, F1, and support for each class.

        Args:
            stage: Evaluation stage name ('val' or 'test'), used as a prefix for logging.
        """
        predictions, labels = zip(*self.val_step_outputs)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        try:
            report_table = self._create_classification_report(labels, predictions)
            self.logger.experiment.log({f"{stage}_classification_report": report_table})
        except Exception as e:
            self.log_.warning(f"Error logging classification report: {str(e)}")
        finally:
            self.val_step_outputs.clear()

    def _create_classification_report(
            self,
            labels: torch.Tensor,
            predictions: torch.Tensor
    ) -> wandb.Table:
        """
        Generate a W&B table containing a detailed classification report.

        Args:
            labels: Ground truth class labels.
            predictions: Predicted class labels.

        Returns:
            A W&B table with per-class precision, recall, F1-score, and support.
        """
        report = classification_report(
            labels.numpy(),
            predictions.numpy(),
            zero_division=0,
            output_dict=True,
            target_names=self.target_names
        )
        report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
        for label_name, metrics in report.items():
            if not isinstance(metrics, dict):
                continue
            report_table.add_data(
                label_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"]
            )
        return report_table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Computes loss and logs relevant metrics, including per-class accuracy.

        Args:
            batch: A tuple of (images, labels) from the training dataloader.
            batch_idx: Index of the current batch.

        Returns:
            The training loss for backpropagation.
        """
        metrics, accuracy_classes = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='train')
        train_loss = metrics.pop('train_loss')
        self.log('train_loss', train_loss, prog_bar=True)
        self.log_dict(metrics, logger=True)
        for i, class_accuracy in enumerate(accuracy_classes):
            self.log(
                f'train_accuracy_class-{self.target_names[i]}',
                round(class_accuracy, 6),
                on_step=False,
                on_epoch=True
            )
        return train_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Computes and logs validation metrics, including per-class accuracy.

        Args:
            batch: A tuple of (images, labels) from the validation dataloader.
            batch_idx: Index of the current batch.

        Returns:
            The validation loss for later aggregation or logging.
        """
        metrics, acc_classes = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='val')
        self.log_dict(metrics, logger=True)
        for i, class_accuracy in enumerate(acc_classes):
            self.log(f'val_accuracy_class-{self.target_names[i]}', round(class_accuracy, 6), on_step=False, on_epoch=True)
        return metrics['val_loss']

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single test step.

        Computes and logs test metrics, including per-class accuracy.

        Args:
            batch: A tuple of (images, labels) from the test dataloader.
            batch_idx: Index of the current batch.

        Returns:
            The test loss for further aggregation or logging.
        """
        metrics, acc_classes = self.calculate_metrics(batch=batch, batch_idx=batch_idx, stage='test')
        self.log_dict(metrics, logger=True)
        for i, class_accuracy in enumerate(acc_classes):
            self.log(f'test_accuracy_class-{self.target_names[i]}', round(class_accuracy, 6), on_step=False, on_epoch=True)
        return metrics['test_loss']

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.

        Logs mean average precision and the full classification report.
        """
        self.log_map_on_epoch_end('val')
        self.log_classification_report('val')

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.

        Logs mean average precision and the full classification report.
        """
        self.log_map_on_epoch_end('test')
        self.log_classification_report('test')

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            A tuple of ([optimizer], [scheduler]) for Lightning to use.
        """
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs)
        return [optimizer], [scheduler]