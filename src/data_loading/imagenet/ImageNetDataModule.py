import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class ImageNetDataModule(pl.LightningDataModule):
    """Downloads the test portion of the ImageNet dataset and provides a dataloader

    TODO: Implement real DataModule
    """

    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        """Initializes the Module

        Args:
            data_dir: path where the downloaded dataset will be stored
            batch_size: batch size used by the dataloader
            num_workers: number of cpu workers for loading the data (unused)
        """

        super(ImageNetDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_dataset = None

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self) -> None:
        """Download ImageNet dataset, here not applicable"""
        # ImageNet(root=self.data_dir, split='train', download=True)
        pass

    def setup(self, stage=None) -> None:
        """Define test dataset"""
        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}", transform=self.transform
            )

    def test_dataloader(self) -> DataLoader:
        """Initialises the test dataloader

        Returns: Dataloader for test dataset
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
