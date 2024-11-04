import os
import wandb
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from common_utils.config import CONFIG

from data_loading.datasets import DataLoaderFactory
from trainings_module import TrainingModule
from models import ModelFactory

wandb.require("core")


class SingleRun:
    """ Run class to execute a training run on a specific model and dataset """
    logs_path = '/media/storagecube/olivers/logs/logs'  # f'{ROOT_DIR}/logs'
    checkpoint_path = '/media/storagecube/olivers/logs/checkpoints'
    dl_collection = DataLoaderFactory()
    model_collection = ModelFactory()
    main_metrics = {
        'multiclass': 'val_accuracy_micro',
        'multilabel': 'val_mAP_micro'
    }

    def __init__(
        self,
        dataset_name,
        model_name,
        devices,
        do_training=True,
        pretrained=False,
        val_transform: dict = None,
        # val_transform_name=None,
        # val_transform_param=None,
        logger=None,
        random_id=420000,
        epochs=None
    ):
        # arguments
        self.log = logger or logging.getLogger("Run")
        self.log.setLevel(logging.DEBUG)
        self.dataset_name = dataset_name
        self.data_config = CONFIG['datasets'][dataset_name]
        self.model_name = model_name
        self.epochs = epochs or self.data_config['epochs']
        self.devices = devices
        self.do_training = do_training
        self.pretrained = pretrained
        # self.val_transform_param = val_transform_param
        # self.val_transform_name = val_transform_name
        # self.val_transform = val_transform
        self.val_transform = val_transform
        self.main_metric = self.main_metrics[self.data_config['task']]
        self.checkpoint_dir = f"{self.checkpoint_path}/{dataset_name}/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # setup
        self.train_dl, self.val_dl, self.test_dl = self.dl_collection.get_dataloader(
            dataset_name=self.dataset_name,
            eval_transform=self.val_transform['transform'],
        )
        self.model = self.model_collection.get_model(
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
        self.logger = self._init_logger()
        self.trainer = self._init_trainer()

    def _get_run_name(self, random_id):
        train_type = 'ST' if self.do_training else 'PT' if self.pretrained else 'NT'
        transform_params_log = f'-{self.val_transform["type"]}-{self.val_transform["param"]}' if train_type != 'ST' else ''
        return f"{self.dataset_name}-{self.model_name}-{train_type}{transform_params_log}-{random_id:06d}"

    def _init_logger(self):
        logger_tags = [
            f"model:{self.model_name}",
            f"data:{self.dataset_name}",
            f"pretrained:{self.pretrained}",
            f"train:{self.do_training}",
            f"val_transform:{self.val_transform['type']}",
            f"val_transform_param:{self.val_transform['param']}",
        ]
        os.makedirs(self.logs_path, exist_ok=True)
        logger = WandbLogger(project=f"Master Thesis",
                             name=self.run_name,
                             group=self.dataset_name,
                             tags=logger_tags,
                             save_dir=self.logs_path,
                             version=f"-{self.run_name}")
        return logger

    def _init_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            monitor=self.main_metric,
            dirpath=self.checkpoint_dir,
            filename='{epoch:02d}-{' + self.main_metric + ':.3f}',
            mode='max'
        )
        trainer = Trainer(
            max_epochs=self.epochs,
            logger=self.logger,
            callbacks=[checkpoint_callback] if self.do_training else [],
            limit_train_batches=CONFIG['limit_train_batches'],
            limit_val_batches=CONFIG['limit_val_batches'],
            limit_test_batches=CONFIG['limit_test_batches'],
            enable_model_summary=False,
        )
        return trainer

    def _load_best_model_from_checkpoint(self):
        best_score = 0
        best_score_checkpoint = ""
        for checkpoint in os.listdir(self.checkpoint_dir):
            if self.main_metric not in checkpoint:
                continue
            score = checkpoint.split('=')[2].replace('.ckpt', '').split('-')[0]
            score = float(score)
            if score > best_score:
                best_score = score
                best_score_checkpoint = checkpoint
        assert best_score_checkpoint != ""
        checkpoint_path = f"{self.checkpoint_dir}/{best_score_checkpoint}"
        self.log.info(f"Loading checkpoint {best_score_checkpoint}")
        model_module = TrainingModule.load_from_checkpoint(
            checkpoint_path,
            model=self.model,
            dataset_name=self.dataset_name,
            dataset_config=self.data_config
        )
        return model_module

    def execute(self):
        if self.do_training:
            self.trainer.fit(self.model_module, self.train_dl, self.val_dl)
        elif self.pretrained is False:
            self.model_module = self._load_best_model_from_checkpoint()
        self.trainer.test(self.model_module, self.test_dl)
        wandb.finish()
        self.log.debug("Run finished")
