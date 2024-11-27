import torch
from torchvision.transforms import Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from common_utils.logger import create_logger
from common_utils.config import CONFIG
from torchvision.transforms import ToTensor

# from models import ModelFactory
from data_loading.imagenet.imagenet_datamodule import ImageNetDataModule
from data_loading.caltech.caltech_datamodule import CaltechDataModule
from data_loading.bigearthnet.BENv2_DataModule import BENv2DataModule
from data_loading.deepglobe.deepglobe_datamodule import DeepglobeDataModule




class DataLoaderFactory:
    """ All DataModules need to be implemented to only require a train and eval transform"""
    log = create_logger("Data Loading")
    # model_collection = ModelFactory()
    data_modules = {
        'imagenet': ImageNetDataModule,
        'caltech': CaltechDataModule,
        'caltech_ft': CaltechDataModule,
        'caltech_120': CaltechDataModule,
        'bigearthnet': BENv2DataModule,
        'rgb_bigearthnet': BENv2DataModule,
        'deepglobe': DeepglobeDataModule,
    }
    dataset_names = list(data_modules.keys())

    def get_datamodule(self, dataset_name, train_transform, eval_transform):
        data_module = self.data_modules[dataset_name]
        if dataset_name in ['caltech_120', 'rgb_bigearthnet']:
            return data_module(train_transform=train_transform,
                               eval_transform=eval_transform,
                               dataset_name=dataset_name)
        return data_module(train_transform=train_transform, eval_transform=eval_transform)

    def get_dataloader(self, dataset_name, train_transform=None, eval_transform=None):
        datamodule = self.get_datamodule(
            dataset_name=dataset_name,
            train_transform=train_transform,
            eval_transform=eval_transform
        )
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()


if __name__ == "__main__":
    from sanity_checks.check_dataloader import print_dataloader_sizes
    print("DATASETS")
    factory = DataLoaderFactory()
    for dataset_name in factory.dataset_names:
        print(f"DATASET: {dataset_name}")
        train_loader, val_loader, test_loader = factory.get_dataloader(dataset_name)
        print_dataloader_sizes(train_loader, val_loader, test_loader)
        print("\n")

