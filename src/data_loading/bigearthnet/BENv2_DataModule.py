from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union


try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from configilm.extra.data_dir import dataset_paths
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from configilm.util import Messages
from common_utils.config import CONFIG


# MINE
DATA_DIRS = dataset_paths['benv2'][1]  # 1 for erde server


class BENv2DataModule(pl.LightningDataModule):
    num_classes = 19
    train_ds: Union[None, BENv2DataSet] = None
    val_ds: Union[None, BENv2DataSet] = None
    test_ds: Union[None, BENv2DataSet] = None
    train_transforms: Optional[Callable] = None
    eval_transforms: Optional[Callable] = None

    def __init__(
        self,
        data_dirs: Mapping[str, Union[str, Path]] = DATA_DIRS,  # MINE
        batch_size: int = CONFIG['batch_size'],  # MINE
        num_workers_dataloader: int = CONFIG['num_workers'],  # MINE
        shuffle: Optional[bool] = True,  # MINE
        max_len: Optional[int] = None,
        pin_memory: Optional[bool] = CONFIG['pin_memory'],  # MINE
        patch_prefilter: Optional[Callable[[str], bool]] = None,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        always_use_default_transforms: bool = True,  # MINE
        dataset_name = 'bigearthnet',
    ):
        """
        This datamodule is designed to work with the BigEarthNet dataset. It is a
        multi-label classification dataset. The dataset is split into train, validation
        and test sets. The datamodule provides dataloaders for each of these sets.

        :param data_dirs: A mapping from file key to file path. Required keys are
            "images_lmdb", "metadata_parquet" and "metadata_snow_cloud_parquet". The "images_lmdb"
            key is used to identify the lmdb file that contains the images. The "metadata_" keys
            are used to identify the parquet files that contain the metadata. The metadata files
            contain information about the images, such as the labels, split and cloud and snow info.
            Note, that the lmdb file is encoded using the RICO-HDL Encoder and contains
            images in the form of safe files.

        :param batch_size: The batch size to use for the dataloaders.

            :default: 16

        :param img_size: The size of the images. Note that this includes the number of
            channels. For example, if the images are RGB images, the size should be
            (3, h, w). See BEN_DataSet.avail_chan_configs for available channel
            configurations.

            :default: (3, 120, 120)

        :param num_workers_dataloader: The number of workers to use for the dataloaders.

            :default: 4

        :param shuffle: Whether to shuffle the data. If None is provided, the data is shuffled
            for training and not shuffled for validation and test.

            :default: None

        :param max_len: The maximum number of images to use. If None or -1 is provided,
            all images are used. Applies per split.

            :default: None

        :param pin_memory: Whether to use pinned memory for the dataloaders. If None is
            provided, it is set to True if a GPU is available and False otherwise.

            :default: None

        :param patch_prefilter: A callable that is used to filter out images. If None is
            provided, no filtering is applied. The callable should take a string as input
            and return a boolean. If the callable returns True, the image is included in
            the dataset, otherwise it is excluded.

            :default: None

        :param train_transforms: A callable that is used to transform the training images.
            If None is provided, the default train transform is used, that consists of
            resizing, horizontal and vertical flipping and normalization.

            :default: None

        :param eval_transforms: A callable that is used to transform the evaluation images.
            If None is provided, the default eval transform is used, that consists of
            resizing and normalization.

            :default: None
        """
        train_transforms = train_transform
        eval_transforms = eval_transform
        super().__init__()

        DATA_CONFIG = CONFIG['datasets'][dataset_name]
        self.img_size = (DATA_CONFIG['input_channels'], DATA_CONFIG['image_size'], DATA_CONFIG['image_size'])
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.num_workers_dataloader = num_workers_dataloader
        self.max_len = max_len
        self.patch_prefilter = patch_prefilter

        self.pin_memory = pin_memory
        if self.pin_memory is None:
            self.pin_memory = True if torch.cuda.is_available() else False

        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.warn(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        # get mean and std
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])
        _default_train_transform = default_train_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
        )
        _default_eval_transform = default_transform(
            img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
        )
        if train_transforms is not None and always_use_default_transforms:
            Messages.warn("Using both default and provided train transform (default first).")
            self.train_transform = Compose([_default_train_transform, train_transforms])
        elif train_transforms is not None:
            self.train_transform = train_transforms
        else:
            Messages.warn("Using default train transform.")
            self.train_transform = _default_train_transform

        if eval_transforms is not None and always_use_default_transforms:
            Messages.warn("Using both default and provided eval transform (default first).")
            self.transform = Compose([_default_eval_transform, eval_transforms])
        elif eval_transforms is not None:
            self.transform = eval_transforms
        else:
            Messages.warn("Using default eval transform.")
            self.transform = _default_eval_transform

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepares the data sets for the specific stage.

        - "fit": train and validation data set
        - "test": test data set
        - None: all data sets

        Prints the time it needed for this operation and other statistics if print_infos
        is set.

        :param stage: None, "fit" or "test"
        """
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BENv2DataSet(
                self.data_dirs,
                split="train",
                transform=self.train_transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )

            self.val_ds = BENv2DataSet(
                self.data_dirs,
                split="validation",
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BENv2DataSet(
                self.data_dirs,
                split="test",
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        Messages.info(sample_info_msg)

    def train_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the train set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the train set
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the validation set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the validation set
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Create a Dataloader according to the specification in the `__init__` call.
        Uses the test set and expects it to be set (e.g. via `setup()` call)

        :return: torch DataLoader for the test set
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )


if __name__ == '__main__':
    data_module = BENv2DataModule(
        dataset_name='rgb_bigearthnet',
    )
    data_module.setup()