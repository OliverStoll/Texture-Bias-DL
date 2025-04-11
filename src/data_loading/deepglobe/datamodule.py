import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from common_utils.config import CONFIG

from data_loading.deepglobe.dataset import DeepGlobeDataset

DATA_CONFIG = CONFIG['datasets']['deepglobe']


class DeepglobeDataModule(pl.LightningDataModule):
    num_cls = 6
    means = [0.4095 * 255, 0.3808 * 255, 0.2836 * 255]
    stds = [0.1509 * 255, 0.1187 * 255, 0.1081 * 255]

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
    """Test the Deepglobe DataModule class and display the min max values for normalization checking."""
    dm = DeepglobeDataModule()
    dm.setup()
    train_dl, val_dl, test_dl = dm.all_dataloader()
    first_batch = next(iter(train_dl))
    print(first_batch[0].shape, first_batch[1].shape)
    for idx in range(10):
        print(first_batch[0][idx].min(), first_batch[0][idx].max())
