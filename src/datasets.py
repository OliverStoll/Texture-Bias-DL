import pandas as pd
from torchvision.models import (ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
                                Swin_T_Weights, ConvNeXt_Tiny_Weights)
from torchvision import transforms
from torchvision.transforms import Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from data_loading.BENv2DataModule import BENv2DataModule
from data_loading.ImageNetDataModule import ImageNetDataModule
from models import ModelCollection
from sanity_checks.check_dataloader import check_dataloader
from utils.config import CONFIG
from utils.logger import create_logger


class DataLoaderCollection:
    log = create_logger("Data Loading")
    model_collection = ModelCollection()

    def __init__(self):
        self.datamodule_getter = {
            "bigearthnet": self._get_datamodule_bigearthnet,
            "imagenet": self._get_datamodule_imagenet,
        }

    def _get_bigearthnet_keys(self, splits_path):
        files = {"train": "all_train.csv", "validation": "all_val.csv", "test": "all_test.csv"}
        keys = {}
        for split, file in files.items():
            keys[split] = pd.read_csv(f"{splits_path}/{file}", header=None, names=["name"]
                                      ).name.to_list()
        return keys

    def _get_datamodule_bigearthnet(self, train_transform, val_transform):
        return BENv2DataModule(
            train_transforms=train_transform,
            eval_transforms=val_transform,
            keys=self._get_bigearthnet_keys(
                splits_path="/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits"),
            image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
            label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
            s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            img_size=CONFIG['datasets']['bigearthnet']['image_size'],
            interpolation_mode=None
        )

    def _get_datamodule_imagenet(self, train_transform, val_transform):
        return ImageNetDataModule(
            train_transforms=train_transform,
            val_transforms=val_transform,
            data_dir=CONFIG['datasets']['imagenet']['path'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            pin_memory=CONFIG['pin_memory'],
            train_val_test_split=CONFIG['datasets']['imagenet']['train_val_test_split'],
        )

    def get_default_transform(self, model_name):
        dataset_config = CONFIG['datasets']['imagenet']
        model = self.model_collection.get_model(model_name, dataset_config=dataset_config,
                                                pretrained=False)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return transform

    def combine_input_with_default_transform(self, model_name, additional_train_transform,
                                             additional_val_transform):
        base_t = self.get_default_transform(model_name=model_name)
        additional_train_transform = base_t if additional_train_transform is None else Compose(
            [base_t, additional_train_transform
         ])
        additional_val_transform = base_t if additional_val_transform is None else Compose(
            [base_t, additional_val_transform
         ])
        return additional_train_transform, additional_val_transform

    def get_dataloader(self, dataset_name, model_name, train_transform=None, val_transform=None):
        self.log.debug(f"Initializing dataloader: [{dataset_name.upper()} | {model_name.upper()}"
                       f"{' | Val_Transform' if val_transform else ''}]")
        if dataset_name == "imagenet":
            train_transform, val_transform = self.combine_input_with_default_transform(
                model_name=model_name,
                additional_train_transform=train_transform,
                additional_val_transform=val_transform,
            )
        datamodule = self.datamodule_getter[dataset_name](
            train_transform=train_transform,
            val_transform=val_transform
        )
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()


if __name__ == "__main__":
    dl_collection = DataLoaderCollection()
    dl_collection.get_default_transform("resnet")
