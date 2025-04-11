import pickle

import lmdb
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_loading.deepglobe.constants import DEEPGLOBE_NAME2IDX


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
