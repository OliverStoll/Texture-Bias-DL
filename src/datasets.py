import pandas as pd
from torchvision.models import (ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
                                Swin_T_Weights, ConvNeXt_Tiny_Weights)
from torchvision import transforms
from torchvision.transforms import Compose

from data_loading.BENv2DataModule import BENv2DataModule
from data_loading.ImageNetDataModule import ImageNetDataModule
from sanity_checks.check_dataloader import check_dataloader
from utils.config import CONFIG
from utils.logger import create_logger








class DataLoaderCollection:
    log = create_logger("Data Loading")

    def __init__(self):
        self.datamodule_getter = {
            "bigearthnet": self._get_datamodule_bigearthnet,
            "imagenet": self._get_datamodule_imagenet,
        }
        self.default_transforms = {
            'resnet': ResNet50_Weights.DEFAULT.transforms(),
            'efficientnet': EfficientNet_B0_Weights.DEFAULT.transforms(),
            'convnext': ConvNeXt_Tiny_Weights.DEFAULT.transforms(),
            'vit': ViT_B_16_Weights.DEFAULT.transforms(),
            'swin': Swin_T_Weights.DEFAULT.transforms(),
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

    def add_default_transform(self, model_name, train_transform, val_transform):
        base_t = self.default_transforms[model_name]
        train_transform = base_t if train_transform is None else Compose([base_t, train_transform])
        val_transform = base_t if val_transform is None else Compose([base_t, val_transform])
        return train_transform, val_transform

    def get_dataloader(self, dataset_name, model_name, train_transform=None, val_transform=None):
        self.log.debug(f"Initializing dataloader: [{dataset_name.upper()} | {model_name.upper()}"
                       f"{' | Val_Transform' if val_transform else ''}]")
        if dataset_name == "imagenet":
            train_transform, val_transform = self.add_default_transform(
                model_name=model_name,
                train_transform=train_transform,
                val_transform=val_transform,
            )
        datamodule = self.datamodule_getter[dataset_name](
            train_transform=train_transform,
            val_transform=val_transform
        )
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()


if __name__ == "__main__":
    dl_collection = DataLoaderCollection()