import os
import wandb
from pytorch_lightning import seed_everything
from common_utils.config import CONFIG
from common_utils.logger import create_logger
from random import randint
from typing import Sequence

from data_loading.datasets import DataLoaderFactory
from models.models import ModelFactory
from transforms.transforms_factory import empty_transform
from utils.logs import mute_logs
from tests.gpu import print_gpu_info
from run_single import SingleRun


class RunManager:
    """
    Manages configuration and execution settings for training and evaluation runs.
    Handles model and dataset selection, device assignment, test run setup, and seeding.
    """

    log = create_logger("Main")
    test_run_batches = 25

    def __init__(
            self,
            eval_transforms: object = None,
            models: Sequence[str] | None = None,
            datasets: Sequence[str] | None = None,
            continue_on_error: bool = True,
            train: bool | None = None,
            pretrained: bool | None = None,
            test_run: bool | int = False,
            device_idx: int | None = None,
            verbose: bool = False,
            epochs: int | None = None,
            seed: int = 42,
            id_digits: int = 8
    ) -> None:
        """
        Initialize the run configuration.

        Args:
            eval_transforms: Transformations to apply during evaluation.
            models: List of model names to use; if None, loaded from environment or defaults.
            datasets: List of dataset names to use; if None, loaded from environment or defaults.
            continue_on_error: Whether to proceed if an error occurs during a run.
            train: Whether to train models.
            pretrained: Whether to use pretrained models.
            test_run: Whether to perform a lightweight test run (bool or int for batch count).
            device_idx: Target GPU device for computation.
            verbose: Enable detailed logging if True.
            epochs: Number of training epochs.
            seed: Random seed for reproducibility.
            id_digits: Number of digits for generating unique run IDs.
        """
        self.test_run = test_run
        self.epochs = epochs
        self.models = models or list(os.getenv("MODEL"))
        if self.models is None:
            self.models = ModelFactory().all_model_names

        self.datasets = datasets or list(os.getenv("DATASET"))
        if self.datasets is None:
            self.datasets = DataLoaderFactory().dataset_names

        self.eval_transforms = eval_transforms or empty_transform
        self.continue_on_error = continue_on_error
        self.train = train
        self.pretrained = pretrained

        self.device = self._determine_device(device_idx)
        self.random_ids = self._init_run_ids(id_digits)

        seed_everything(seed)
        if not verbose:
            mute_logs()
        self._log_start()

        if test_run:
            test_batches = self.test_run_batches if isinstance(test_run, bool) else test_run
            self._change_config_for_test_run(test_run_batches=test_batches)


    @staticmethod
    def _init_run_ids(id_digits: int) -> list:
        """ Generate random IDs for each run. """
        max_id = 10 ** id_digits - 1
        return [randint(0, max_id) for _ in range(max_id)]

    def _warn_if_gpu_utilized(self):
        """ Log a warning if GPU is already utilized. """
        utilization = print_gpu_info()
        if int(utilization) > 0:
            self.log.warning(f"GPU {self.device} is already in use")

    def _determine_device(self, device: int | None) -> list:
        """ Determine the device to use for training. """
        if device is not None:
            return [device]
        devices = [int(os.getenv("MY_DEVICE", "-1"))]
        if devices == [-1]:
            devices = [int(input("ENTER GPU INDEX: "))]
        self.log.info(f"Using GPU {devices}")
        self._warn_if_gpu_utilized()
        return devices

    def _log_start(self) -> None:
        """Logs the initial configuration of the run."""
        transform_types = {t['type'] for t in self.eval_transforms}
        transform_str = ', '.join(transform_types) if transform_types != {None} else ''

        self.log.info(
            f"\n\n\tSTARTING RUNS:"
            f"\n\t\tMODELS:      {', '.join(self.models)}"
            f"\n\t\tDATASETS:    {', '.join(self.datasets)}"
            f"\n\t\tTRAIN:       {'AUTO' if self.train is None else self.train}"
            f"\n\t\tPRETRAINED:  {'AUTO' if self.pretrained is None else self.pretrained}"
            f"\n\t\tTRANSFORM:   {transform_str}\n\n"
        )

    def _log_run_time_estimation(
        self,
        number_of_runs: int,
        avg_minutes_per_run: int = 5,
        test_run_factor: float = 0.04,
        upper_bound_factor: float = 3.0
    ) -> None:
        """ Estimate and log the expected run time based on the number of runs. """
        self.log.info(f"Starting {number_of_runs} runs on GPU: {self.device}")
        if self.train is False:
            lower_bound = (number_of_runs * avg_minutes_per_run) / 60
            if self.test_run:
                lower_bound *= test_run_factor
            upper_bound = lower_bound * upper_bound_factor
            self.log.info(f"This could take ~ {lower_bound:.0f}-{upper_bound:.0f} hours")


    def _change_config_for_test_run(self, test_run_batches: int | None = None) -> None:
        """
        Adjusts global configuration for a lightweight test run.

        Args:
            test_run_batches: Number of batches to use for each phase (train/val/test).
        """
        self.log.warning(f"Test-Run with only {test_run_batches} batches each")
        CONFIG['limit_train_batches'] = test_run_batches * 2
        CONFIG['limit_val_batches'] = test_run_batches
        CONFIG['limit_test_batches'] = test_run_batches


    def execute_single_run(
        self,
        dataset: str,
        model: str,
        val_transform_dict: dict,
        run_idx: int,
        train: bool,
        pretrained: bool
    ) -> None:
        """
        Executes a single training or evaluation run with the specified configuration.

        Args:
            dataset: Name of the dataset to use.
            model: Name of the model to run.
            val_transform_dict: Dictionary containing evaluation transform metadata.
            run_idx: Index of the current run in the sweep.
            train: Whether to train the model.
            pretrained: Whether to use a pretrained model.
        """
        number_of_total_runs = len(self.models) * len(self.datasets) * len(self.eval_transforms)
        try:
            self.log.info(
                f"[{run_idx}/{number_of_total_runs}] Starting Run "
                f"[{dataset}"
                f" | {model}"
                f" | {'train' if train else 'pretrain' if pretrained else 'eval'}"
                f" | {val_transform_dict['type']}"
                f" | {val_transform_dict['param']}]"
            )
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
                self.log.info("Continuing with next Run")
            else:
                raise e
        finally:
            wandb.finish()

    def execute_all_runs(self) -> None:
        """
        Executes all combinations of models, datasets, and evaluation transforms.
        Logs device, estimated runtime, and delegates each configuration to `execute_single_run`.
        """
        number_of_runs = len(self.models) * len(self.datasets) * len(self.eval_transforms)
        self._log_run_time_estimation(number_of_runs)
        run_idx = 0
        for dataset in self.datasets:
            pretrained = self.pretrained if self.pretrained is not None else (dataset == "imagenet")
            train = self.train if self.train is not None else not pretrained
            for model in self.models:
                for eval_transform in self.eval_transforms:
                    run_idx += 1
                    self.execute_single_run(dataset, model, eval_transform, run_idx, train, pretrained)


if __name__ == '__main__':
    run_models = ['efficientnet', 'swin', 'convnext', 'resnet']
    run_datasets = DataLoaderFactory().dataset_names

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        continue_on_error=False,
        train=True,
        pretrained=True,
        test_run=True,
        device_idx=3,
    )
    run_manager.execute_all_runs()
    wandb.alert(title="All runs finished", text="Completed successfully")
