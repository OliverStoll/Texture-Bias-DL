import os
import wandb
import logging
import warnings
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from utils.config import ROOT_DIR, CONFIG
from utils.logger import create_logger

from datasets import DataLoaderCollection
from training_module import TrainingModule
from models import ModelCollection
from _util_code._mute_logs import mute_logs
from sanity_checks.check_gpu import print_gpu_info
from grid_shuffle_transform import GridShuffleTransform

warnings.filterwarnings("ignore",
                        message="Average precision score for one or more classes was `nan`. Ignoring these classes in weighted-average")


class Run:
    """ Run class to execute a training run on a specific model and dataset

        Abbreviations: dl = dataloader
                       grid_shuffle = the number of grid squares for width and height (4 -> 4x4)
    """
    dl_collection = DataLoaderCollection()
    model_collection = ModelCollection()

    def __init__(self, dataset_name, model_name, do_training=True, pretrained=False, val_transform=None,
                 grid_shuffle=None):
        # arguments
        self.log = logging.getLogger("Run")
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.do_training = do_training
        self.pretrained = pretrained
        # setup
        self.data_config = CONFIG['datasets'][dataset_name]
        self.run_name = (f"{model_name}-{dataset_name}-{'PT' if pretrained else 'ST'}"
                         f"{f'-{grid_shuffle}' if grid_shuffle else ''}")
        dataloader_tuple = self.dl_collection.get_dataloader(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             is_pretrained=pretrained,
                                                             val_transform=val_transform)
        self.train_dl, self.val_dl, self.test_dl = dataloader_tuple
        self.model = self.model_collection.get_model(model_name=self.model_name,
                                                     dataset_config=self.data_config,
                                                     pretrained=self.pretrained)
        self.model_module = TrainingModule(model=self.model,
                                           dataset_name=dataset_name,
                                           dataset_config=self.data_config)
        logger_tags = [f"model:{model_name}", f"data:{dataset_name}",
                       'pretrained:true' if pretrained else 'pretrained:false',
                       f'grid_shuffle:{grid_shuffle}' if (
                               grid_shuffle and val_transform) else 'grid_shuffle:false']
        os.makedirs(f'{ROOT_DIR}logs', exist_ok=True)
        self.logger = WandbLogger(project=f"Master Thesis",
                                  name=self.run_name,
                                  group=dataset_name,
                                  tags=logger_tags,
                                  save_dir=f'{ROOT_DIR}logs',
                                  version=f"-{self.run_name}")
        self.profiler = SimpleProfiler(dirpath=f'{ROOT_DIR}logs/profiler',
                                       filename=f'{dataset_name}-{model_name}',
                                       extended=True)
        self.trainer = Trainer(max_epochs=CONFIG['epochs'],
                               logger=self.logger,
                               profiler=self.profiler,
                               devices=DEVICES,
                               limit_train_batches=CONFIG['limit_train_batches'],
                               limit_val_batches=CONFIG['limit_val_batches'],
                               limit_test_batches=CONFIG['limit_test_batches'],
                               enable_model_summary=False)


    def _get_dataloader(self, dataset, model, is_pretrained, train_transforms=None,
                        val_transforms=None):
        dl_collection = DataLoaderCollection()
        return dl_collection.get_dataloader(dataset, model, is_pretrained, train_transforms,
                                            val_transforms)

    def load_checkpoint(self, metric='val_mAP_micro'):
        """checkpoint_path = '/media/storagecube/olivers/logs/checkpoints/convnext/epoch=09-val_mAP_micro=0.870.ckpt'"""
        checkpoint_dir = f"{CONFIG['checkpoint_dir']}/{self.model_name}"
        best_score = 0
        best_score_checkpoint = ""
        for checkpoint in os.listdir(checkpoint_dir):
            if metric not in checkpoint:
                continue
            score = checkpoint.split('=')[2].replace('.ckpt', '')
            score = float(score)
            if score > best_score:
                best_score = score
                best_score_checkpoint = checkpoint

        assert best_score_checkpoint != ""
        checkpoint_path = f"{checkpoint_dir}/{best_score_checkpoint}"
        self.log.info(f"Loading checkpoint {best_score_checkpoint}")
        self.model_module = TrainingModule.load_from_checkpoint(checkpoint_path, model=self.model,
                                                                dataset_name=self.dataset_name,
                                                                dataset_config=self.data_config)

    def execute(self):
        self.log.info(f"Starting Run for {self.model_name} on {self.dataset_name} ({DEVICES})")
        if self.pretrained:
            self.log.info("Starting pretrained Test")
            self.trainer.test(self.model_module, self.test_dl)
        if self.pretrained is False and self.do_training is False:
            self.log.info("Starting checkpointed Test")
            self.load_checkpoint()
            self.trainer.test(self.model_module, self.test_dl)
        if self.do_training:
            self.log.info("Starting Training")
            self.trainer.fit(self.model_module, self.train_dl, self.val_dl)
            self.trainer.test(self.model_module, self.test_dl)
        wandb.finish()
        self.log.info("Run finished")


class RunManager:
    log = create_logger("Main")
    test_run_batches = 500

    def __init__(self, models=None, datasets=None, continue_on_error=True, grid_sizes=None,
                 train='auto', pretrained='auto', test_run=False):
        self.models = models or list(os.getenv("MODEL"))
        self.datasets = datasets or list(os.getenv("DATASET"))
        self.continue_on_error = continue_on_error
        self.grid_sizes = grid_sizes
        self.train = train
        self.pretrained = pretrained
        self.grid_shuffle_transforms = self._get_grid_shuffle_transforms(grid_sizes)
        self._check_input()
        seed_everything(42)
        mute_logs()
        if test_run:
            self.change_config_for_test_run()

    def _get_grid_shuffle_transforms(self, grid_sizes):
        """create a list of tuples, with grid size and the equivalent transform"""
        grid_sizes = grid_sizes or [1]
        grid_shuffle_transforms = [{
            'grid_size': grid_size,
            'transform': GridShuffleTransform(grid_size=grid_size)} for grid_size in grid_sizes]
        return grid_shuffle_transforms

    def _check_input(self):
        assert self.models is not None, "Models not specified"
        assert self.datasets is not None, "Datasets not specified"

    def _log_start(self):
        self.log.info(f"\n\n\n\tSTARTING RUNS FOR {self.models} ON {self.datasets} "
                      f"(PT: {self.pretrained} - Grid sizes: {self.grid_sizes}\n\n")

    def change_config_for_test_run(self):
        self.log.warning(f"Test-Run with only {self.test_run_batches} batches each")
        CONFIG['limit_train_batches'] = self.test_run_batches
        CONFIG['limit_val_batches'] = self.test_run_batches
        CONFIG['limit_test_batches'] = self.test_run_batches

    def execute_runs(self):
        for dataset in self.datasets:
            pretrained = (dataset == 'imagenet') if self.pretrained == 'auto' else self.pretrained
            train = not pretrained if self.train == 'auto' else self.train
            for model in self.models:
                for val_transform_dict in self.grid_shuffle_transforms:
                    try:
                        run = Run(dataset_name=dataset,
                                  model_name=model,
                                  do_training=train,
                                  pretrained=pretrained,
                                  val_transform=val_transform_dict['transform'],
                                  grid_shuffle=val_transform_dict['grid_size'])
                        run.execute()
                        del run
                    except Exception as e:
                        self.log.error(f"Run failed for {model} on {dataset}", e)
                        if self.continue_on_error:
                            self.log.info(f"Continung with next Run")
                        else:
                            raise e


def determine_gpu():
    print_gpu_info()
    devices = [int(os.getenv("MY_DEVICE", "-1"))]
    if devices is [-1]:
        devices = [int(input("ENTER GPU INDEX: "))]
    return devices


if __name__ == '__main__':
    DEVICES = determine_gpu()
    run_models = ['resnet', 'vit', 'efficientnet', 'swin', 'convnext']
    run_datasets = ['bigearthnet', 'imagenet']
    run_grid_sizes = range(1, 21)


    run_datasets = ['imagenet']  # noqa
    # run_models = ['convnext']  # noqa
    # run_grid_sizes = [13]  # noqa
    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        grid_sizes=run_grid_sizes,
        continue_on_error=False,
        train=False,
        # test_run=True,
    )
    run_manager.execute_runs()