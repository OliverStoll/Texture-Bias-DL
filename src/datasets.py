import torch
from torchvision.transforms import Compose
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils.logger import create_logger
from utils.config import CONFIG
from torchvision.transforms import ToTensor

from models import ModelFactory
from data_loading.imagenet.imagenet_datamodule import ImageNetDataModule
from data_loading.caltech.caltech_datamodule import CaltechDataModule
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from configilm.extra.data_dir import dataset_paths



class DataLoaderFactory:
    log = create_logger("Data Loading")
    model_collection = ModelFactory()
    ben_image_size = CONFIG['datasets']['bigearthnet']['image_size']
    ben_image_channels = CONFIG['datasets']['bigearthnet']['input_channels']
    ben_mapping = dataset_paths['benv2'][1]  # 1 for erde server


    def __init__(self):
        self.datamodule_getter = {
            "bigearthnet": self._get_datamodule_bigearthnet,
            "imagenet": self._get_datamodule_imagenet,
            "caltech": self._get_datamodule_caltech,
        }
        self.dataset_names = list(self.datamodule_getter.keys())

    def _get_datamodule_bigearthnet(self, train_transform, val_transform):
        return BENv2DataModule(
            train_transforms=train_transform,
            eval_transforms=val_transform,
            data_dirs=self.ben_mapping,
            img_size=(self.ben_image_channels, self.ben_image_size, self.ben_image_size),
        )

    def _get_datamodule_imagenet(self, train_transform, val_transform):
        return ImageNetDataModule(
            train_transforms=train_transform,
            eval_transforms=val_transform,
        )

    def _get_datamodule_caltech(self, train_transform, val_transform):
        return CaltechDataModule(
            train_transforms=train_transform,
            eval_transforms=val_transform,
        )

    def _get_datamodule_deepglobe(self, train_transform, val_transform):
        NotImplementedError("DeepGlobe not implemented yet")

    def get_datamodule(self, dataset_name, train_transform, val_transform):
        return self.datamodule_getter[dataset_name](train_transform, val_transform)

    def get_imagenet_default_transform(self, model_name):
        if model_name in self.ignore_imagenet_transform:
            return ToTensor()  # TODO: normalization?
        imagenet_config = CONFIG['datasets']['imagenet']
        model = self.model_collection.get_model(
            model_name,
            dataset_config=imagenet_config,
            pretrained=False
        )
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return transform

    def combine_input_with_default_transform(self, model_name, input_train_transform, input_val_transform):
        base_transform = self.get_imagenet_default_transform(model_name=model_name)
        combined_train_transform = Compose([base_transform, input_train_transform])
        combined_val_transform = Compose([base_transform, input_val_transform])
        return combined_train_transform, combined_val_transform

    def get_dataloader(self, dataset_name, train_transform=None, val_transform=None):
        datamodule = self.get_datamodule(
            dataset_name=dataset_name,
            train_transform=train_transform,
            val_transform=val_transform
        )
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()


if __name__ == "__main__":
    dl_collection = DataLoaderFactory().get_dataloader('imagenet', 'resnet')
    # get first img tensor of train dataloader
    train_dl = dl_collection[0]
    batch = next(iter(train_dl))
    tensor = batch[0][0]
    # save to file
    torch.save(tensor, '/home/olivers/colab-master-thesis/output/test_data/Imagenet_tensor.pt')
    exit()
    check_all(*dl_collection)
