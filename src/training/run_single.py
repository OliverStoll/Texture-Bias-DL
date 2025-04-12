from typing import Any, Sequence
import os
import wandb
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from common_utils.config import CONFIG

from data_loading.datasets import DataLoaderFactory
from training.module import TrainingModule
from models.models import ModelFactory

wandb.require("core")



class SingleRun:
    """
    Run class to execute a training run on a specific model and dataset.
    Handles data loading, model instantiation, training configuration, and logging setup.
    """

    dataloader_factory = DataLoaderFactory()
    model_factory = ModelFactory()
    main_metrics = {
        'multiclass': 'val_accuracy_macro',
        'multilabel': 'val_mAP_macro'
    }

    def __init__(
            self,
            dataset_name: str,
            model_name: str,
            devices: Sequence[str],
            do_training: bool = True,
            pretrained: bool = False,
            val_transform: dict | None = None,
            logger: logging.Logger | None = None,
            random_id: int = 420000,
            epochs: int | None = None,
            logs_path: str = '/media/storagecube/olivers/logs/logs',
            checkpoint_path: str = '/media/storagecube/olivers/logs/checkpoints',
            early_stopping_patience: int = 5,
            wandb_project_name: str = 'feature_bias_classification',
    ) -> None:
        """
        Initialize a training run with the specified dataset, model, and configuration.

        Args:
            dataset_name: Name of the dataset to be used.
            model_name: Name of the model architecture to be trained.
            devices: List of device identifiers (e.g., ["cuda:0"]).
            do_training: If False, only evaluation will be performed.
            pretrained: Whether to use pretrained weights.
            val_transform: Dictionary containing evaluation transforms.
            logger: Logger instance to use; defaults to a new one if not provided.
            random_id: Random identifier for tracking runs.
            epochs: Number of epochs to train for; if None, uses dataset default.
            logs_path: Path to store log files.
            checkpoint_path: Path to store model checkpoints.
            early_stopping_patience: Number of epochs to wait for improvement before stopping.
            wandb_project_name: Name of the Weights & Biases project for logging.
        """
        self.log = logger or logging.getLogger("Run")
        self.log.setLevel(logging.DEBUG)
        self.logs_path = logs_path
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.data_config = CONFIG['datasets'][dataset_name]
        self.model_name = model_name
        self.epochs = epochs or self.data_config['epochs']
        self.devices = devices
        self.do_training = do_training
        self.pretrained = pretrained
        self.val_transform = val_transform
        self.main_metric = self.main_metrics[self.data_config['task']]
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = f"{self.checkpoint_path}/{dataset_name}/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_dl, self.val_dl, self.test_dl = self.dataloader_factory.get_dataloader(
            dataset_name=self.dataset_name,
            eval_transform=self.val_transform['transform'],
        )

        self.model = self.model_factory.get_model(
            model_name=self.model_name,
            dataset_config=self.data_config,
            pretrained=self.pretrained
        )

        self.model_module = TrainingModule(
            model=self.model,
            dataset_name=self.dataset_name,
            dataset_config=self.data_config
        )

        self.run_name = self._get_run_name(random_id)
        self.logger = self._init_wandb_logger(wandb_project_name)
        self.trainer = self._init_trainer()


    def _get_run_name(self, run_id: int) -> str:
        """
        Generate a unique run name based on dataset, model, training type,
        transform parameters in the case of evaluation, and a numeric run ID.

        Args:
            run_id: The unique identifier for this run.

        Returns:
            A formatted string representing the run name.
        """
        if self.do_training:
            train_type = "ST"
        elif self.pretrained:
            train_type = "PT"
        else:
            train_type = "NT"
        if train_type == "PT" or train_type == "NT":
            transform_suffix = f"-{self.val_transform['type']}-{self.val_transform['param']}"
        else:
            transform_suffix = ""
        return f"{self.dataset_name}-{self.model_name}-{train_type}{transform_suffix}-{run_id:06d}"

    def _init_wandb_logger(self, wandb_project_name) -> WandbLogger:
        """
        Initialize a Weights & Biases logger for the current run.

        Creates the log directory if it does not exist and sets up run-specific metadata
        for model, dataset, and training configuration.

        Args:
            wandb_project_name: The name of the Weights & Biases project.
        """
        os.makedirs(self.logs_path, exist_ok=True)
        logger_tags = [
            f"model:{self.model_name}",
            f"data:{self.dataset_name}",
            f"pretrained:{self.pretrained}",
            f"train:{self.do_training}",
            f"val_transform:{self.val_transform['type']}",
            f"val_transform_param:{self.val_transform['param']}",
        ]
        logger = WandbLogger(
            project=wandb_project_name,
            name=self.run_name,
            group=self.dataset_name,
            tags=logger_tags,
            save_dir=self.logs_path,
            version=f"-{self.run_name}"
        )
        return logger

    def _init_trainer(self) -> Trainer:
        """
        Initialize the PyTorch Lightning Trainer with run-specific configuration.

        Returns:
            A configured Trainer instance.
        """
        callbacks = self._init_callbacks()

        trainer = Trainer(
            max_epochs=self.epochs,
            logger=self.logger,
            callbacks=callbacks,
            limit_train_batches=CONFIG['limit_train_batches'],
            limit_val_batches=CONFIG['limit_val_batches'],
            limit_test_batches=CONFIG['limit_test_batches'],
            enable_model_summary=False,
        )

        return trainer

    def _init_callbacks(self) -> list[Callback]:
        """
        Initialize training callbacks including model checkpointing and early stopping.

        Returns:
            A list of callbacks to be used by the Trainer. Empty if training is disabled.
        """
        if self.do_training is False:
            return []

        checkpoint_callback = ModelCheckpoint(
            monitor=self.main_metric,
            dirpath=self.checkpoint_dir,
            filename='{epoch:02d}-{' + self.main_metric + ':.3f}',
            mode='max'
        )
        early_stop_callback = EarlyStopping(
            monitor=self.main_metric,
            patience=self.early_stopping_patience,
            verbose=True,
            mode='max'
        )

        return [checkpoint_callback, early_stop_callback]

    def _select_best_model_checkpoint(self) -> str | None:
        """
        Select the checkpoint file with the highest score.

        This is done by parsing the filenames in the checkpoint directory,
        using the pattern: {epoch:02d}-<metric>=<score>.ckpt.

        Returns:
            The filename of the best model checkpoint.
        """
        best_score = 0
        best_checkpoint_filename = None
        for checkpoint_filename in os.listdir(self.checkpoint_dir):
            if self.main_metric not in checkpoint_filename:
                continue
            parts = checkpoint_filename.replace('.ckpt', '').split('=')
            score_str = parts[-1].split('-')[0]
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_checkpoint_filename = checkpoint_filename

        return best_checkpoint_filename

    def _load_best_model_from_checkpoint(self) -> TrainingModule:
        """
        Load the best-performing validation metric model checkpoint.

        Returns:
            An instance of the TrainingModule loaded from the best checkpoint.
        """
        best_checkpoint_filename = self._select_best_model_checkpoint()
        if best_checkpoint_filename is None:
            raise FileNotFoundError("No checkpoint found in the directory.")
        best_checkpoint_path = f"{self.checkpoint_dir}/{best_checkpoint_filename}"
        self.log.info(f"Loading checkpoint {best_checkpoint_filename}")
        model_module = TrainingModule.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path,
            model=self.model,
            dataset_name=self.dataset_name,
            dataset_config=self.data_config
        )
        return model_module

    def execute(self) -> None:
        """
        Execute the training or evaluation run.

        - If training is enabled, trains the model.
        - If evaluating using pretrained weights, loads the model from the model factory
        - If evaluation self-trained model, loads the best checkpoint.
        In all cases, evaluates the model on the test dataset.
        """
        if self.do_training:
            self.trainer.fit(self.model_module, self.train_dl, self.val_dl)
        elif self.pretrained is False:
            self.model_module = self._load_best_model_from_checkpoint()
        self.trainer.test(self.model_module, self.test_dl)
        wandb.finish()
        self.log.debug("Run finished")
