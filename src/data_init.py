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





def get_datamodule_bigearthnet(train_transform, val_transform):
    def get_bigearthnet_keys(dir):
        files = {"train": "all_train.csv", "validation": "all_val.csv", "test": "all_test.csv"}
        keys = {}
        for split, file in files.items():
            keys[split] = pd.read_csv(f"{dir}/{file}", header=None, names=["name"]
                                      ).name.to_list()
        return keys

    return BENv2DataModule(
        train_transforms=train_transform,
        eval_transforms=val_transform,
        keys=get_bigearthnet_keys(dir="/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits"),
        image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
        label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
        s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        img_size=CONFIG['datasets']['bigearthnet']['image_size'],
        interpolation_mode=None
    )


def get_datamodule_imagenet(train_transform, val_transform):
    return ImageNetDataModule(
        train_transforms=train_transform,
        val_transforms=val_transform,
        data_dir=CONFIG['datasets']['imagenet']['path'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        train_val_test_split=CONFIG['datasets']['imagenet']['train_val_test_split'],
    )


class DataLoaderCollection:
    log = create_logger("Data Loading")
    datamodule_init_fns = {
            "bigearthnet": get_datamodule_bigearthnet,
            "imagenet": get_datamodule_imagenet
        }
    default_transforms = {
            'resnet': ResNet50_Weights.DEFAULT.transforms(),
            'efficientnet': EfficientNet_B0_Weights.DEFAULT.transforms(),
            'convnext': ConvNeXt_Tiny_Weights.DEFAULT.transforms(),
            'vit': ViT_B_16_Weights.DEFAULT.transforms(),
            'swin': Swin_T_Weights.DEFAULT.transforms(),
        }

    def __init__(self):
        pass

    def get_dataloader(self, dataset_name, model_name, is_pretrained, train_transform=None, val_transform=None):
        self.log.debug(f"Initializing dataloader: [{dataset_name.upper()} | {model_name.upper()}{' | Pretrained' if is_pretrained else ''}]")
        dm_init_function = self.datamodule_init_fns[dataset_name]
        if dataset_name == "imagenet":
            base_t = self.default_transforms[model_name]
            train_transform = base_t if train_transform is None else Compose([base_t, train_transform])
            val_transform = base_t if val_transform is None else Compose([base_t, val_transform])
        datamodule = dm_init_function(train_transform=train_transform, val_transform=val_transform)
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

    def get_all_dataloaders(self, dataset_names, model_names):
        dataloaders = {}
        for dataset in dataset_names:
            dataloaders[dataset] = {}
            for model_name in model_names:
                is_pretrained = (dataset == "imagenet")
                train_dl, val_dl, test_dl = self.get_dataloader(dataset, model_name, is_pretrained)
                dataloaders[dataset][model_name] = {"train": train_dl, "val": val_dl, "test": test_dl}
        return dataloaders


if __name__ == "__main__":
    dl_collection = DataLoaderCollection()
    models = ['resnet', 'efficientnet', 'convnext', 'vit', 'swin']
    datasets = ['bigearthnet']
    models = ['resnet']
    all_dl = dl_collection.get_all_dataloaders(dataset_names=datasets, model_names=models)
    dl = all_dl['bigearthnet']['resnet']
    check_dataloader(dl['train'], dl['val'], dl['test'])
    print(all_dl)