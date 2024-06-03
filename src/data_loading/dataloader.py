import pandas as pd
from torchvision import transforms

from data_loading.bigearthnet.BENv2DataModule import BENv2DataModule
from data_loading.imagenet.ImageNetDataModule import ImageNetDataModule
from sanity_checks.check_dataloader import check_dataloader, analyze_dataloader_class_balance
from utils.config import CONFIG
from utils.logger import create_logger


log = create_logger("Data Loading")


def get_datamodule_bigearthnet(train_transform, val_transform):
    def get_bigearthnet_keys(dir):
        keys = {}
        files = {"train": "all_train.csv", "validation": "all_val.csv", "test": "all_test.csv"}
        for split, file in files.items():
            file_path = f"{dir}/{file}"
            split_list = pd.read_csv(file_path, header=None, names=["name"]).name.to_list()
            keys[split] = split_list
        return keys

    log.debug(f"Using {'default' if val_transform is None else str(val_transform)} val-transform "
              f"{'(and no extra train transform)' if train_transform is None else ''}")
    if val_transform is None:
        mean_std = pd.read_csv("/home/olivers/colab-master-thesis/src/data_loading/bigearthnet/all_bands_mean_std.csv")
        mean, std = mean_std["mean"].to_list(), mean_std["std"].to_list()
        val_transform = transforms.Compose([
            transforms.Normalize(mean, std),
            transforms.Resize(112),
        ])
    if train_transform is None:
        train_transform = val_transform

    return BENv2DataModule(
        train_transforms=train_transform,
        eval_transforms=val_transform,
        keys=get_bigearthnet_keys(dir="/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits"),
        image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
        label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
        s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
    )


def get_datamodule_imagenet(train_transform, val_transform):
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        log.debug(f"Using default val-transform{' (and no extra train transform)' if train_transform is None else ''}")
    else:
        log.debug(f"Using provided val-transform: \n{val_transform}")
    if train_transform is None:
        train_transform = val_transform
    return ImageNetDataModule(
        train_transforms=train_transform,
        val_transforms=val_transform,
        data_dir=CONFIG['datasets']['imagenet']['path'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        train_val_test_split=CONFIG['datasets']['imagenet']['train_val_test_split'],
    )


def get_dataloader(dataset_name, train_transform=None, val_transform=None):
    """ Get dataloaders for a specific dataset. Uses the default transforms if none are provided."""
    log.info(f"SETTING UP DATALOADER: {dataset_name}")
    match dataset_name:
        case "bigearthnet":
            datamodule = get_datamodule_bigearthnet(train_transform, val_transform)
        case "imagenet":
            datamodule = get_datamodule_imagenet(train_transform, val_transform)
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    datamodule.setup()

    _train_loader = datamodule.train_dataloader()
    _val_loader = datamodule.val_dataloader()
    _test_loader = datamodule.test_dataloader()

    # check_dataloader(_train_loader, _val_loader, _test_loader)
    if CONFIG['check_class_balance']:
        analyze_dataloader_class_balance(_train_loader, _val_loader, _test_loader)

    return _train_loader, _val_loader, _test_loader


def get_all_dataloader(dataset_names, train_transform_dict=None, val_transform_dict=None):
    """ Get dataloaders for multiple datasets. Uses the default transforms if none are provided."""
    dataloaders = {}
    for dataset in dataset_names:
        train_transform = train_transform_dict[dataset] if train_transform_dict is not None else None
        val_transform = val_transform_dict[dataset] if val_transform_dict is not None else None
        dataloaders[dataset] = get_dataloader(dataset, train_transform, val_transform)


if __name__ == "__main__":
    get_all_dataloader(['bigearthnet', 'imagenet'])