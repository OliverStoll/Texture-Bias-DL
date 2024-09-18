from torchvision.transforms import Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils.logger import create_logger
from utils.config import CONFIG
# impor totensor
from torchvision.transforms import ToTensor

from models import ModelCollection
from sanity_checks.check_dataloader import check_dataloader
from data_loading.ImageNetDataModule import ImageNetDataModule
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from configilm.extra.data_dir import dataset_paths



class DataLoaderCollection:
    log = create_logger("Data Loading")
    model_collection = ModelCollection()
    ben_image_size = CONFIG['datasets']['bigearthnet']['image_size']
    ben_image_channels = CONFIG['datasets']['bigearthnet']['input_channels']
    ignore_imagenet_transform = ['flexivit']

    def __init__(self):
        self.datamodule_getter = {
            "bigearthnet": self._get_datamodule_bigearthnet,
            "imagenet": self._get_datamodule_imagenet,
        }

    def _get_datamodule_bigearthnet(self, train_transform, val_transform):
        benv2_mapping = dataset_paths['benv2'][1]  # 1 for erde server
        return BENv2DataModule(
            data_dirs=benv2_mapping,
            batch_size=CONFIG['batch_size'],
            num_workers_dataloader=CONFIG['num_workers'],
            pin_memory=CONFIG['pin_memory'],
            img_size=(self.ben_image_channels, self.ben_image_size, self.ben_image_size),
            train_transforms=train_transform,
            eval_transforms=val_transform,
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

    def get_imagenet_default_transform(self, model_name):
        if model_name in self.ignore_imagenet_transform:
            return ToTensor()  # TODO: normalization?
        imagenet_config = CONFIG['datasets']['imagenet']
        model = self.model_collection.get_model(model_name, dataset_config=imagenet_config,
                                                pretrained=False)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return transform

    def combine_input_with_default_transform(self, model_name, additional_train_transform,
                                             additional_val_transform):
        base_t = self.get_imagenet_default_transform(model_name=model_name)
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
    dl_collection.get_imagenet_default_transform("resnet")
