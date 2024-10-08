import os
import wandb
import logging
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.config import ROOT_DIR, CONFIG
from utils.logger import create_logger
from random import randint

from datasets import DataLoaderFactory
from training import TrainingModule
from models import ModelFactory
from transforms import empty_transforms
from util_code.logs import mute_logs
from sanity_checks.check_gpu import print_gpu_info

wandb.require("core")


class Run:
    """ Run class to execute a training run on a specific model and dataset """
    logs_path = f'{ROOT_DIR}/logs'  # '/media/storagecube/olivers/logs'
    checkpoint_path = '/media/storagecube/olivers/logs/checkpoints'
    dl_collection = DataLoaderFactory()
    model_collection = ModelFactory()
    main_metrics = {
        'multiclass': 'val_accuracy_micro',
        'multilabel': 'val_mAP_micro'
    }

    def __init__(self, dataset_name, model_name, devices, do_training=True, pretrained=False,
                 val_transform=None, val_transform_name=None, val_transform_param=None,
                 logger=None, random_id=420000, epochs=None):
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
        self.val_transform_param = val_transform_param
        self.val_transform_name = val_transform_name
        self.val_transform = val_transform
        self.main_metric = self.main_metrics[self.data_config['task']]
        self.checkpoint_dir = f"{self.checkpoint_path}/{dataset_name}/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # setup
        self.train_dl, self.val_dl, self.test_dl = self.dl_collection.get_dataloader(
            dataset_name=self.dataset_name,
            eval_transform=self.val_transform
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
        self.run_name = self._init_run_name(random_id)
        self.logger = self._init_logger()
        self.trainer = self._init_trainer()

    def _init_run_name(self, random_id):
        train_type = 'ST' if self.do_training else 'PT' if self.pretrained else 'NT'
        transform_params_log = f'-{self.val_transform_name}-{self.val_transform_param}' if train_type != 'ST' else ''
        return f"{self.dataset_name}-{self.model_name}-{train_type}{transform_params_log}-{random_id:06d}"

    def _init_logger(self):
        logger_tags = [
            f"model:{self.model_name}",
            f"data:{self.dataset_name}",
            f"pretrained:{self.pretrained}",
            f"train:{self.do_training}",
            f"val_transform:{self.val_transform_name}",
            f"val_transform_param:{self.val_transform_param}"
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
            dirpath=f'{self.checkpoint_path}/{self.model_name}',
            filename='{epoch:02d}-{'+self.main_metric+':.3f}',
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

    def load_checkpoint(self):
        checkpoint_dir = f"{CONFIG['checkpoint_dir']}/{self.model_name}"
        best_score = 0
        best_score_checkpoint = ""
        for checkpoint in os.listdir(checkpoint_dir):
            if self.main_metric not in checkpoint:
                continue
            score = checkpoint.split('=')[2].replace('.ckpt', '').split('-')[0]
            score = float(score)
            if score > best_score:
                best_score = score
                best_score_checkpoint = checkpoint
        assert best_score_checkpoint != ""
        checkpoint_path = f"{checkpoint_dir}/{best_score_checkpoint}"
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
            self.model_module = self.load_checkpoint()
        self.trainer.test(self.model_module, self.test_dl)
        wandb.finish()
        self.log.debug("Run finished")


class RunManager:
    log = create_logger("Main")
    test_run_batches = 25

    def __init__(
            self,
            eval_transforms=None,
            models=None,
            datasets=None,
            continue_on_error=True,
            train=None,
            pretrained=None,
            test_run=False,
            device=None,
            verbose=False,
            epochs=None,
    ):
        self.epochs = epochs
        self.models = models or list(os.getenv("MODEL"))
        if self.models is None:
            self.models = ModelFactory().all_model_names
        self.datasets = datasets or list(os.getenv("DATASET"))
        if self.datasets is None:
            self.datasets = DataLoaderFactory().dataset_names

        self.eval_transforms = eval_transforms or empty_transforms
        # settings
        self.continue_on_error = continue_on_error
        self.train = train
        self.pretrained = pretrained
        # self.device = self.determine_gpu() if device is None else [device]
        self.device = None
        self.random_ids = [randint(0, 999999) for _ in range(1000)]
        utilization = print_gpu_info()
        if int(utilization) > 0:
            self.log.warning(f"GPU {self.device} is already in use")
        seed_everything(42)
        if verbose is False:
            mute_logs()
        self._log_start()
        if test_run:
            test_run_batches = self.test_run_batches if test_run is True else test_run
            self.change_config_for_test_run(test_run_batches=test_run_batches)

    def determine_gpu(self):
        devices = [int(os.getenv("MY_DEVICE", "-1"))]
        if devices == [-1]:
            devices = [int(input("ENTER GPU INDEX: "))]
        self.log.info(f"Using GPU {devices}")
        return devices

    def _log_start(self):
        transform_types = {t['type'] for t in self.eval_transforms}
        self.log.info(f"\n\n\tSTARTING RUNS:"
                      f"\n\t\tMODELS:      {', '.join(self.models)}"
                      f"\n\t\tDATASETS:    {', '.join(self.datasets)}"
                      f"\n\t\tTRAIN:       {'AUTO' if self.train is None else self.train}"
                      f"\n\t\tPRETRAINED:  {'AUTO' if self.pretrained is None else self.pretrained}"
                      f"\n\t\tTRANSFORM:   {', '.join(transform_types) if transform_types != {None} else ''}\n\n")

    def change_config_for_test_run(self, test_run_batches=None):
        self.log.warning(f"Test-Run with only {test_run_batches} batches each")
        CONFIG['limit_train_batches'] = test_run_batches * 2
        CONFIG['limit_val_batches'] = test_run_batches
        CONFIG['limit_test_batches'] = test_run_batches


    def execute_single_run(self, dataset, model, val_transform_dict, run_idx, train, pretrained):
        number_of_runs = len(self.models) * len(self.datasets) * len(self.eval_transforms)
        try:
            self.log.info(f"[{run_idx}/{number_of_runs}] Starting Run "
                          f"[{dataset}"
                          f" | {model}"
                          f" | {'train' if train else 'pretrain' if pretrained else 'eval'}"
                          f" | {val_transform_dict['type']}"
                          f" | {val_transform_dict['param']}]")
            run = Run(dataset_name=dataset,
                      model_name=model,
                      do_training=train,
                      pretrained=pretrained,
                      val_transform=val_transform_dict['transform'],
                      val_transform_name=val_transform_dict['type'],
                      val_transform_param=val_transform_dict['param'],
                      logger=self.log,
                      devices=self.device,
                      random_id=self.random_ids[run_idx],
                      epochs=self.epochs)
            run.execute()

        except Exception as e:
            self.log.error(f"Run failed for {model} on {dataset}", e)
            if self.continue_on_error:
                self.log.info(f"Continung with next Run")
            else:
                raise e
        finally:
            wandb.finish()

    def execute_runs(self):
        number_of_runs = len(self.models) * len(self.datasets) * len(self.eval_transforms)
        self.log.info(f"Starting {number_of_runs} runs on GPU: {self.device}")
        if self.train is False:
            self.log.info(f"This could take ~ {number_of_runs / 12:.0f}-{number_of_runs / 4:.0f} hours")
        run_idx = 0
        for dataset in self.datasets:
            pretrained = (dataset == 'imagenet') if self.pretrained is None else self.pretrained
            train = (not pretrained) if self.train is None else self.train
            if train and pretrained:
                self.log.warning("Training and Pretraining at the same time")
            for model in self.models:
                for val_transform_dict in self.eval_transforms:
                    run_idx += 1
                    self.execute_single_run(dataset, model, val_transform_dict, run_idx, train, pretrained)


if __name__ == '__main__':
    run_models = ['efficientnet', 'swin', 'convnext', 'resnet']
    run_datasets = DataLoaderFactory().dataset_names
    run_datasets = ['caltech']  # noqa
    # run_models = ['vit']  # noqa

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        continue_on_error=False,
        train=True,
        pretrained=True,
        test_run=True,
        device=3,
    )
    run_manager.execute_runs()
    wandb.alert(title="All runs finished", text="Completed successfully")
