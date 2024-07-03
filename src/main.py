import os

from torch import nn
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
import logging
import warnings

# utils
from utils.config import ROOT_DIR, CONFIG
from utils.logger import create_logger

# local
from data_init import DataLoaderCollection
from lightning_module import ModelModule
from model_init import ModelCollection
from _util_code import _seed_everything, _mute_logs  # noqa (run helper code)
from sanity_checks.check_gpu import print_gpu_info

warnings.filterwarnings("ignore", message="Average precision score for one or more classes was `nan`. Ignoring these classes in weighted-average")


class Run:
    """ Run class to execute a training run on a specific model and dataset

        Abbreviations: dl = dataloader
    """
    log = logging.getLogger("Run")
    dl_collection = DataLoaderCollection()
    model_collection = ModelCollection()

    def __init__(self, dataset_name, model_name, train=True, pretrained=False):
        # arguments
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.train = train
        self.pretrained = pretrained
        # setup
        self.data_config = CONFIG['datasets'][dataset_name]
        dataloader_tuple = self.dl_collection.get_dataloader(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             is_pretrained=pretrained)
        self.train_dl, self.val_dl, self.test_dl = dataloader_tuple
        self.model = self.model_collection.get_model(model_name=self.model_name,
                                                     dataset_config=self.data_config,
                                                     pretrained=self.pretrained)
        self.model_module = ModelModule(model=self.model,
                                        dataset_name=dataset_name,
                                        dataset_config=self.data_config)
        logger_tags = [f"model:{model_name}", f"data:{dataset_name}",
                       'pretrained:true' if pretrained else 'pretrained:false']
        self.logger = WandbLogger(project=f"Master Thesis",
                                  name=f"{model_name}-{dataset_name}{'-PT' if pretrained else ''}",
                                  group=dataset_name,
                                  tags=logger_tags,
                                  save_dir=f'{ROOT_DIR}logs')
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

    def _get_dataloader(self, dataset, model, is_pretrained, train_transforms=None, val_transforms=None):
        dl_collection = DataLoaderCollection()
        return dl_collection.get_dataloader(dataset, model, is_pretrained, train_transforms, val_transforms)

    def execute(self):
        self.log.info(f"Starting Run for {self.model_name} on {self.dataset_name} ({DEVICES})")
        if self.pretrained:
            self.log.info("Starting pretrained Test")
            self.trainer.test(self.model_module, self.test_dl)
        if self.train:
            self.log.info("Starting Training")
            self.trainer.fit(self.model_module, self.train_dl, self.val_dl)
            self.trainer.test(self.model_module, self.test_dl)
        wandb.finish()
        self.log.info("Run finished")


def run_all(models=None, datasets=None, continue_on_error=True, pretrained=None):
    models = models or list(os.getenv("MODEL"))
    datasets = datasets or list(os.getenv("DATASET"))
    assert models is not None, "Models not specified"
    assert datasets is not None, "Datasets not specified"
    log.info(f"\n\n\n\t\tSTARTING RUNS FOR {models} ON {datasets} (pretrained: {pretrained})\n\n")
    for dataset in datasets:
        for model in models:
            try:
                pretrained = False if pretrained is False else (dataset == 'imagenet')
                run = Run(dataset_name=dataset,
                          model_name=model,
                          train=not pretrained,
                          pretrained=pretrained)
                run.execute()
                del run
            except Exception as e:
                if continue_on_error:
                    log.error(f"Run failed for {model} on {dataset}", e)
                else:
                    raise e


def determine_gpu():
    # determine gpu
    print_gpu_info()
    DEVICES = [int(os.getenv("MY_DEVICE", "-1"))]
    if DEVICES is [-1]:
        DEVICES = [int(input("ENTER GPU INDEX: "))]
    return DEVICES


if __name__ == '__main__':
    log = create_logger("Main")
    DEVICES = determine_gpu()
    _models = ['vit', 'swin', 'convnext', 'resnet', 'efficientnet']
    _datasets = ['bigearthnet', 'imagenet']
    _pretrained = False
    _continue_on_error = False

    _models = ['swin', 'vit']
    _datasets = ['bigearthnet', 'imagenet']
    run_all(_models, _datasets, _continue_on_error, _pretrained)

