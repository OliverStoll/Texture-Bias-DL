from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from common_utils.config import CONFIG
from typing import Callable

from data_loading.splitting import stratify_split

DATA_CONFIG = CONFIG['datasets']['imagenet']


class ImageNetDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the ImageNet dataset.

    Handles dataset instantiation, preprocessing transformations,
    and construction of train/val/test DataLoaders.
    """

    default_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
            self,
            train_transform: Callable | None = None,
            eval_transform: Callable | None = None,
            also_use_default_transforms: bool = True,
            data_dir: str = DATA_CONFIG['path'],
            batch_size: int = CONFIG['batch_size'],
            num_workers: int = CONFIG['num_workers'],
            pin_memory: bool = CONFIG['pin_memory'],
            train_val_test_split: list[int] = CONFIG['train_val_test_split'],
            dataloader_timeout: int = CONFIG['dataloader_timeout'],
            shuffle: bool = False,
    ):
        """
        Initialize the ImageNet data module.

        Args:
            train_transform: Optional transformation for training images.
            eval_transform: Optional transformation for validation/test images.
            also_use_default_transforms: Whether to prepend default transforms to custom ones.
            data_dir: Directory where the dataset is stored.
            batch_size: Batch size for all DataLoaders.
            num_workers: Number of worker processes for loading data.
            pin_memory: Whether to pin memory in data loaders.
            train_val_test_split: List of integers or floats for train/val/test sizes.
            dataloader_timeout: Timeout in seconds for data loading.
            shuffle: Whether to shuffle training data.
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_test_split = train_val_test_split
        self.dataloader_timeout = dataloader_timeout

        self.train_transform, self.eval_transform = self.get_correct_transforms(
            also_use_default_transforms, train_transform, eval_transform
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_correct_transforms(
        self,
        use_default: bool,
        train_transform: Callable | None,
        eval_transform: Callable | None
    ) -> tuple[Callable, Callable]:
        """
        Compose final train and eval transforms by optionally prepending a default base transform.

        Args:
            use_default: Whether to include the default transformation before the user-defined one.
            train_transform: Optional user-defined transform for training data.
            eval_transform: Optional user-defined transform for validation/test data.

        Returns:
            A tuple containing (train_transform, eval_transform).
        """
        if not use_default:
            return train_transform, eval_transform

        base_transform = self.default_transformation

        final_train_transform = (
            transforms.Compose([base_transform, train_transform])
            if train_transform else base_transform
        )

        final_eval_transform = (
            transforms.Compose([base_transform, eval_transform])
            if eval_transform else base_transform
        )

        return final_train_transform, final_eval_transform

    def get_dataset(self, transform) -> datasets.ImageFolder:
        """
        Get the ImageNet dataset with the specified transformation.

        Args:
            transform: Transformation to apply to the dataset.

        Returns:
            The ImageNet dataset.
        """
        return datasets.ImageFolder(self.data_dir, transform=transform)

    def setup(self, stage: str | None = None) -> None:
        """
        Prepare the train, validation, and test dataset splits using stratified sampling.

        The dataset is split by label distribution using the configured proportions.
        The same index partitions are applied to separately transformed copies of the dataset
        to allow distinct train and eval pipelines.
        """
        full_dataset_train = self.get_dataset(transform=self.train_transform)
        full_dataset_val = self.get_dataset(transform=self.eval_transform)
        indices = list(range(len(full_dataset_train)))
        labels = np.array([full_dataset_train.targets[i] for i in indices])

        train_size, val_size, test_size = self.train_val_test_split
        train_indices, val_indices, test_indices = stratify_split(
            indices, labels, train_size, val_size, test_size
        )
        self.train_dataset = Subset(full_dataset_train, train_indices)
        self.val_dataset = Subset(full_dataset_val, val_indices)
        self.test_dataset = Subset(full_dataset_val, test_indices)

    def get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """
        Construct a DataLoader for the given dataset with standard parameters.

        Args:
            dataset: The dataset to load.
            shuffle: Whether to shuffle the data each epoch.

        Returns:
            A configured DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.dataloader_timeout,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training DataLoader with shuffling enabled.
        """
        return self.get_dataloader(self.train_dataset, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        """
        Return the validation DataLoader with shuffling disabled.
        """
        return self.get_dataloader(self.val_dataset, shuffle=False)


    def test_dataloader(self) -> DataLoader:
        """
        Return the test DataLoader with shuffling disabled.
        """
        return self.get_dataloader(self.test_dataset, shuffle=False)
