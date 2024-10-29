import lmdb
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common_utils.config import CONFIG

from sanity_checks.check_dataloader import check_dataloader


DEEPGLOBE_COLOR2CLASS = {
    (0, 255, 255): "urban_land",
    (255, 255, 0): "agricultural_land",
    (255, 0, 255): "rangeland",
    (0, 255, 0): "forest_land",
    (0, 0, 255): "water",
    (255, 255, 255): "barren_land",
    (0, 0, 0): "unknown",
}

DEEPGLOBE_IDX2NAME = {
    0: "urban_land",
    1: "agricultural_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
}

DEEPGLOBE_NAME2IDX = {
    "urban_land": 0,
    "agricultural_land": 1,
    "rangeland": 2,
    "forest_land": 3,
    "water": 4,
    "barren_land": 5,
}

DATA_CONFIG = CONFIG['datasets']['deepglobe']


class DeepGlobeDataset(Dataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None, split='train'):
        """
        Args:
            lmdb_path      : path to the LMDB file for efficiently loading the patches.
            csv_path       : path to a csv file containing the patch names
                             that will make up this split
            transform      : specifies the image transform mode which determines the
                             augmentations to be applied to the image
        """
        self.env = None
        self.split = split
        self.lmdb_path = lmdb_path
        self.patch_names = self.read_csv(csv_path)
        self.transform = transform
        self.labels = self.read_labels(labels_path, self.patch_names)

    def read_csv(self, csv_data):
        return pd.read_csv(csv_data, header=None).to_numpy()[:, 0]

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        string_labels = df_subset.labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def convert_to_multihot(self, labels):
        multihot = np.zeros(6)
        indices = [DEEPGLOBE_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )
        patch_name = self.patch_names[idx]
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
        patch = pickle.loads(byteflow)
        label = self.labels[idx]
        patch = self.transform(patch) if self.transform is not None else patch
        return patch, label, idx

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class DeepglobeDataModule(pl.LightningDataModule):
    num_cls = 6
    means = [0.4095, 0.3808, 0.2836]
    stds = [0.1509, 0.1187, 0.1081]

    def __init__(self,
                 train_transform=None,
                 eval_transform=None,
                 also_use_default_transforms: bool = True,
                 image_size=DATA_CONFIG['image_size'],
                 lmdb_path=DATA_CONFIG['path'],
                 labels_path=DATA_CONFIG['labels_path'],
                 csv_paths=DATA_CONFIG['csv_paths'],
                 batch_size=CONFIG['batch_size'],
                 num_workers=CONFIG['num_workers'],
                 pin_memory=CONFIG['pin_memory'],
                 timeout=CONFIG['dataloader_timeout'],
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.image_size = image_size
        self.train_transforms, self.eval_transforms = self.get_correct_transforms(
            also_use_default_transforms, train_transform, eval_transform
        )
        self.lmdb_path = lmdb_path
        self.labels_path = labels_path
        self.csv_paths = csv_paths
        self.setup_complete = False
        self.trainset = None
        self.valset = None
        self.testset = None

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.stds),
            transforms.Resize((self.image_size, self.image_size)),
        ])

    def get_correct_transforms(self, also_use_default_transforms, train_transform,
                               eval_transform):
        if not also_use_default_transforms:
            return train_transform, eval_transform
        base_transform = self._get_default_transform()
        if train_transform is not None:
            final_train_transform = transforms.Compose([base_transform, train_transform])
        else:
            final_train_transform = base_transform
        if eval_transform is not None:
            final_eval_transform = transforms.Compose([base_transform, eval_transform])
        else:
            final_eval_transform = base_transform
        return final_train_transform, final_eval_transform

    def get_dataset(self, transform, split):
        csv_path = self.csv_paths[split]
        return DeepGlobeDataset(self.lmdb_path, csv_path, self.labels_path, transform, split)

    def setup(self, stage=None):
        if self.setup_complete:
            return
        self.trainset = self.get_dataset(transform=self.train_transforms, split='train')
        self.valset = self.get_dataset(transform=self.eval_transforms, split='validation')
        self.testset = self.get_dataset(transform=self.eval_transforms, split='test')
        self.setup_complete = True

    @staticmethod
    def collate_fn(batch):
        images, labels, _ = zip(*batch)
        images = torch.stack(images)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        return [images, labels]

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.trainset, shuffle=True)

    def val_dataloader(self, drop_last=False):
        return self.get_dataloader(self.valset, shuffle=False)

    def test_dataloader(self, drop_last=False):
        return self.get_dataloader(self.testset, shuffle=False)

    def all_dataloader(self):
        if not self.setup_complete:
            self.setup()
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


if __name__ == "__main__":
    dm = DeepglobeDataModule()
    dm.setup()
    train_dl, val_dl, test_dl = dm.all_dataloader()
    first_batch = next(iter(train_dl))
    print()
