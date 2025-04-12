from typing import Callable
from collections import Counter
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import pytorch_lightning as pl
from common_utils.config import CONFIG

from data_loading.caltech.dataset import CaltechRGB101
from data_loading.splitting import SubsetWithLabels


class CaltechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for the Caltech RGB 101 dataset.

    Handles dataset preparation and DataLoader construction for training,
    validation, and testing. Supports exclusion of specific classes and
    composition of default and custom image transforms.
    """

    download = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dataset_name: str = 'caltech',
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        also_use_default_transforms: bool = True,
        batch_size: int = CONFIG['batch_size'],
        num_workers: int = CONFIG['num_workers'],
        pin_memory: bool = CONFIG['pin_memory'],
        train_val_test_split: tuple[float, float, float] = CONFIG['train_val_test_split'],
        dataloader_timeout: int = CONFIG['dataloader_timeout'],
        excluded_class_idx: int | None = 20,
        seed: int = CONFIG['seed'],
    ):
        """
        Initialize the Caltech data module.

        Args:
            dataset_name: Name key for dataset configuration in CONFIG.
            train_transform: Optional custom transform for training.
            eval_transform: Optional custom transform for validation and test.
            also_use_default_transforms: Whether to prepend default transforms to the custom ones.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory in data loaders.
            train_val_test_split: Fractional split of train, val, and test data.
            dataloader_timeout: Timeout (in seconds) for data loading.
            excluded_class_idx: Optional class index to exclude from the dataset.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.excluded_class_idx = excluded_class_idx
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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_default_transform(self) -> Compose:
        """
        Construct the default image transformation pipeline.

        Returns:
            Compose: A torchvision Compose object with resizing, tensor conversion, and normalization.
        """
        return Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_correct_transforms(
            self,
            use_defaults: bool,
            train_transform: Compose | Callable | None,
            eval_transform: Compose | Callable | None
    ) -> tuple[Compose, Compose]:
        """
        Compose the final transformation pipelines for training and evaluation.

        If `use_defaults` is True, the default preprocessing is prepended to the user-provided transforms.

        Args:
            use_defaults: Whether to prepend default transforms to the input transforms.
            train_transform: Optional user-defined training transform.
            eval_transform: Optional user-defined evaluation transform.

        Returns:
            A tuple of (train_transform, eval_transform) composed with or without default transforms.
        """
        if not use_defaults:
            return train_transform, eval_transform

        base_transform = self._get_default_transform()

        final_train_transform = (
            Compose([base_transform, train_transform])
            if train_transform else base_transform
        )

        final_eval_transform = (
            Compose([base_transform, eval_transform])
            if eval_transform else base_transform
        )

        return final_train_transform, final_eval_transform

    def get_dataset(self, transform: Callable | None = None) -> Dataset:
        """
        Load and optionally filter the CaltechRGB101 dataset to retain only the top-N classes.

        If `top_n` is specified, the dataset is filtered to include only the most frequent
        classes, excluding `excluded_class_idx` if set, and labels are remapped to [0, ..., N-1].

        Args:
            transform: Optional image transform to apply.

        Returns:
            A dataset instance, either full or class-filtered.
        """
        dataset = CaltechRGB101(root=self.data_dir, transform=transform)

        if self.top_n is None:
            return dataset

        filtered_indices = self._get_filtered_indices(dataset)
        remapped_labels, class_names = self._get_remapped_labels(dataset, filtered_indices)

        filtered_dataset = SubsetWithLabels(dataset, filtered_indices, remapped_labels)
        filtered_dataset.classes = class_names

        return filtered_dataset

    def _get_filtered_indices(self, dataset: Dataset) -> list[int]:
        """
        Identify indices corresponding to the top-N most frequent classes,
        excluding the specified class index if applicable.

        Args:
            dataset: The full dataset.

        Returns:
            Filtered list of indices to retain.
        """
        valid_indices = [
            i for i, label in enumerate(dataset.y)
            if label != self.excluded_class_idx
        ]

        class_counts = Counter(dataset.y[i] for i in valid_indices)
        self._top_classes = [cls for cls, _ in class_counts.most_common(self.top_n)]

        return [
            i for i, label in enumerate(dataset.y)
            if label in self._top_classes
        ]

    def _get_remapped_labels(self, dataset: Dataset, indices: list[int]) -> tuple[list[int], list[str]]:
        """
        Remap labels of the filtered dataset to a dense range and collect class names.

        Args:
            dataset: The original dataset.
            indices: The selected subset indices.

        Returns:
            A tuple of (remapped labels, class names for the retained classes).
        """
        class_mapping = {original: new for new, original in enumerate(self._top_classes)}
        new_labels = [class_mapping[dataset.y[i]] for i in indices]
        class_names = [dataset.categories[cls] for cls in self._top_classes]
        return new_labels, class_names

    def setup(self, stage=None) -> None:
        """
        Prepare the train, validation, and test datasets.

        Applies the respective transforms and performs a random split based on the configured
        fractions. Ensures reproducibility via a fixed random seed.
        """
        train_base = self.get_dataset(transform=self.train_transforms)
        eval_base = self.get_dataset(transform=self.eval_transforms)

        train_subset, val_subset, test_subset = random_split(
            train_base,
            self.train_val_test_split,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.train_dataset = Subset(train_base, train_subset.indices)
        self.val_dataset = Subset(eval_base, val_subset.indices)
        self.test_dataset = Subset(eval_base, test_subset.indices)

    def get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """
        Construct a DataLoader for the given dataset.

        Args:
            dataset: The dataset to load.
            shuffle: Whether to shuffle the data at each epoch.

        Returns:
            A configured DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            timeout=self.dataloader_timeout,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def get_setup_dataloader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Run setup and return all three data loaders: train, validation, and test.

        Returns:
            A tuple containing the train, val, and test DataLoaders.
        """
        self.setup()
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


if __name__ == '__main__':
    """Test the CaltechDataModule class and save the first image of each batch."""
    import os
    import torch
    test_output_dir = "output/test_data/caltech"
    train_dl, _, _ = CaltechDataModule(num_workers=1).get_setup_dataloader()
    print(len(train_dl))
    os.makedirs(test_output_dir, exist_ok=True)
    for i, (imgs, _) in enumerate(train_dl):
        img = imgs[0]
        img_path = os.path.join(test_output_dir, f"caltech_{i * 32}.png")
        transforms.ToPILImage()(img).save(img_path)

