import pandas as pd

from data_loading.bigearthnet.BENv2DataModule import BENv2DataModule
from data_loading.imagenet.ImageNetDataModule import ImageNetDataModule
from sanity_checks.sanity_checks import sanity_check_dataloader

from utils.config import CONFIG


def get_bigearthnet_keys():
    dir_path = "/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits"
    files = {
        "train": "all_train.csv",
        "validation": "all_val.csv",
        "test": "all_test.csv"
    }
    keys = {}
    for split, file in files.items():
        file_path = f"{dir_path}/{file}"
        split_list = pd.read_csv(file_path, header=None, names=["name"]).name.to_list()
        keys[split] = split_list

    return keys



def get_datamodule_bigearthnet():
    keys = get_bigearthnet_keys()

    datamodule = BENv2DataModule(
        image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
        label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
        s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
        keys=keys,
        num_workers=CONFIG['num_workers'],
    )
    return datamodule


def get_datamodule_imagenet():
    datamodule = ImageNetDataModule(
        data_dir=CONFIG['datasets']['imagenet']['path'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        train_val_test_split=CONFIG['datasets']['imagenet']['train_val_test_split'],
    )
    return datamodule


def get_dataloader(dataset_name):
    match dataset_name:
        case "bigearthnet":
            datamodule = get_datamodule_bigearthnet()
        case "imagenet":
            datamodule = get_datamodule_imagenet()
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # sanity checks
    sanity_check_dataloader(train_loader, val_loader, test_loader)


    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("TEST")