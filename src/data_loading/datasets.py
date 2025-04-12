from common_utils.logger import create_logger
from pytorch_lightning import LightningDataModule
from typing import Callable, Type, Tuple
from logging import Logger

from data_loading.imagenet.datamodule import ImageNetDataModule
from data_loading.caltech.datamodule import CaltechDataModule
from data_loading.bigearthnet.BENv2_DataModule import BENv2DataModule
from data_loading.deepglobe.datamodule import DeepglobeDataModule


class DataLoaderFactory:
    """
    Factory for instantiating DataModules with standardized train/eval transform signatures.

    Ensures that all DataModules follow the expected interface and handles special dataset-specific
    initialization requirements.
    """

    log: Logger = create_logger("Data Loading")
    data_modules: dict[str, Type[LightningDataModule]] = {
        'imagenet': ImageNetDataModule,
        'caltech': CaltechDataModule,
        'caltech_ft': CaltechDataModule,
        'caltech_120': CaltechDataModule,
        'bigearthnet': BENv2DataModule,
        'rgb_bigearthnet': BENv2DataModule,
        'deepglobe': DeepglobeDataModule,
    }
    dataset_names: list[str] = list(data_modules.keys())

    def get_datamodule(
        self,
        dataset_name: str,
        train_transform: Callable | None,
        eval_transform: Callable | None
    ) -> LightningDataModule:
        """
        Instantiate the appropriate DataModule based on dataset name.

        Args:
            dataset_name: Name of the dataset.
            train_transform: Transform to apply to training data.
            eval_transform: Transform to apply to validation/test data.

        Returns:
            An instance of the selected LightningDataModule.
        """
        if dataset_name not in self.data_modules:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        module_cls = self.data_modules[dataset_name]
        module_params = {
            'train_transform': train_transform,
            'eval_transform': eval_transform,
        }
        if dataset_name in {'caltech_120', 'rgb_bigearthnet'}:
            module_params['dataset_name'] = dataset_name

        return module_cls(**module_params)  # noqa: ignore[call-arg]

    def get_dataloader(
        self,
        dataset_name: str,
        train_transform: Callable | None = None,
        eval_transform: Callable | None = None
    ) -> Tuple:
        """
        Instantiate and prepare the DataModule, then return its DataLoaders.

        Args:
            dataset_name: Name of the dataset.
            train_transform: Optional transform for training data.
            eval_transform: Optional transform for validation/test data.

        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader).
        """
        datamodule = self.get_datamodule(
            dataset_name=dataset_name,
            train_transform=train_transform,
            eval_transform=eval_transform
        )
        datamodule.setup()  # noqa: ignore[call-arg]  (None is passed to setup)
        return (
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader()
        )

if __name__ == "__main__":
    from tests.dataloader import print_dataloader_sizes
    print("DATASETS")
    factory = DataLoaderFactory()
    for dataset_name_ in factory.dataset_names:
        print(f"DATASET: {dataset_name_}")
        train_loader, val_loader, test_loader = factory.get_dataloader(dataset_name_)
        print_dataloader_sizes(train_loader, val_loader, test_loader)
        print("\n")

