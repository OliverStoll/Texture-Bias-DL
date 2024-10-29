from typing import List

import torch
from collections import Counter
from torch.utils.data import Subset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import Caltech101
import pytorch_lightning as pl
from common_utils.config import CONFIG

to_tensor = transforms.ToTensor()



class SubsetWithLabels(Subset):
    def __init__(self, dataset, indices, new_labels):
        super().__init__(dataset, indices)
        self.new_labels = new_labels

    def __getitem__(self, idx):
        data, _ = super().__getitem__(idx)
        new_label = self.new_labels[idx]
        return data, new_label

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # print(indices)
        return [self.__getitem__(i) for i in indices]

    def __len__(self):
        return len(self.indices)


class CaltechRGB101():
    def __init__(self, root, transform=None, download=False):
        self.transform = transform
        self.dataset = Caltech101(root=root, transform=None, download=download)
        self.y = self.dataset.y
        self.categories = self.dataset.categories

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.transform(image) if self.transform else image
        return image, label


class CaltechDataModule(pl.LightningDataModule):
    download = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    excluded_class = 20

    def __init__(
        self,
        dataset_name='caltech',
        train_transform=None,
        eval_transform=None,
        also_use_default_transforms: bool = True,  # MINE
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        train_val_test_split=CONFIG['train_val_test_split'],
        dataloader_timeout=CONFIG['dataloader_timeout'],
        seed=CONFIG['seed'],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.dataloader_timeout = dataloader_timeout
        self.seed = seed
        self.data_config = CONFIG['datasets'][dataset_name]
        self.top_n = self.data_config['top_n']
        self.data_dir = self.data_config['path']
        self.image_size = self.data_config['image_size']
        self.train_transforms, self.eval_transforms = self.get_correct_transforms(
            also_use_default_transforms, train_transform, eval_transform
        )
        # datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_correct_transforms(self, also_use_default_transforms, input_train_transform,
                               input_eval_transform):
        if not also_use_default_transforms:
            return input_train_transform, input_eval_transform
        base_transform = self._get_default_transform()
        if input_train_transform is not None:
            combined_train_transform = transforms.Compose([base_transform, input_train_transform])
        else:
            combined_train_transform = base_transform
        if input_eval_transform is not None:
            combined_eval_transform = transforms.Compose([base_transform, input_eval_transform])
        else:
            combined_eval_transform = base_transform
        return combined_train_transform, combined_eval_transform

    def get_dataset(self, transform=None):
        dataset = CaltechRGB101(
            root=self.data_dir,
            transform=transform,
        )
        if self.top_n is None:
            return dataset
        filtered_indices = [i for i, target in enumerate(dataset.y) if target != self.excluded_class]
        class_counts = Counter([dataset.y[i] for i in filtered_indices])
        top_classes = [cls for cls, _ in class_counts.most_common(self.top_n)]
        filtered_indices = [i for i, target in enumerate(dataset.y) if target in top_classes]
        class_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(top_classes)}
        new_targets = [class_to_new_idx[dataset.y[i]] for i in filtered_indices]
        filtered_dataset = SubsetWithLabels(dataset, filtered_indices, new_targets)
        filtered_dataset.classes = [dataset.categories[cls] for cls in top_classes]
        return filtered_dataset

    def setup(self, stage=None) -> None:
        train_dataset = self.get_dataset(transform=self.train_transforms)
        eval_dataset = self.get_dataset(transform=self.eval_transforms)

        # Split the dataset into train, val, test
        train_subset, val_subset, test_subset = random_split(
            train_dataset, self.train_val_test_split,
            generator=torch.Generator().manual_seed(self.seed)
        )
        self.train_dataset = Subset(train_dataset, train_subset.indices)
        self.val_dataset = Subset(eval_dataset, val_subset.indices)
        self.test_dataset = Subset(eval_dataset, test_subset.indices)

    def get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            timeout=self.dataloader_timeout,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def get_setup_dataloader(self):
        self.setup()
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()



if __name__ == '__main__':
    train_dl, _, _ = CaltechDataModule(num_workers=1).get_setup_dataloader()
    batch = next(iter(train_dl))
    print(len(train_dl))
