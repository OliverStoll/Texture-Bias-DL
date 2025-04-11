import os
import wandb
from pytorch_lightning import seed_everything
from common_utils.config import CONFIG
from common_utils.logger import create_logger
from random import randint

from data_loading.datasets import DataLoaderFactory
from models import ModelFactory
from transforms.transforms_factory import empty_transforms
from utils.logs import mute_logs
from tests.gpu import print_gpu_info
from run_single import SingleRun


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
        self.test_run = test_run
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
        self.random_ids = [randint(0, 999999) for _ in range(100000)]
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
            run = SingleRun(
                dataset_name=dataset,
                model_name=model,
                do_training=train,
                pretrained=pretrained,
                val_transform=val_transform_dict,
                logger=self.log,
                devices=self.device,
                random_id=self.random_ids[run_idx],
                epochs=self.epochs
            )
            run.execute()

        except Exception as e:
            self.log.error(f"Run failed for {model} on {dataset}", e)
            if self.continue_on_error:
                self.log.info(f"Continung with next Run")
            else:
                raise e
        finally:
            wandb.finish()

    def execute_multiple_runs(self):
        number_of_runs = len(self.models) * len(self.datasets) * len(self.eval_transforms)
        self.log.info(f"Starting {number_of_runs} runs on GPU: {self.device}")
        if self.train is False:
            time_lower_bound = number_of_runs / 12
            if self.test_run:
                time_lower_bound /= 25
            self.log.info(
                f"This could take ~ {time_lower_bound:.0f}-{time_lower_bound * 3:.0f} hours"
            )
        run_idx = 0
        for dataset in self.datasets:
            pretrained = (dataset == 'imagenet') if self.pretrained is None else self.pretrained
            train = (not pretrained) if self.train is None else self.train
            if train and pretrained:
                self.log.warning("Training and Pretraining at the same time")
            for model in self.models:
                for val_transform_dict in self.eval_transforms:
                    run_idx += 1
                    self.execute_single_run(
                        dataset, model, val_transform_dict, run_idx, train, pretrained
                    )


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
    run_manager.execute_multiple_runs()
    wandb.alert(title="All runs finished", text="Completed successfully")
