from typing import Optional, Callable

from PIL import Image
from torchvision.datasets import Caltech101


class CaltechRGB101:
    """
    Wrapper for the Caltech101 dataset with fault-tolerant loading of non-RGB images.

    Ensures all images are in RGB mode and allows optional image transformations.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False):
        """
        Initialize the CaltechRGB101 dataset.

        Args:
            root: Root directory for the dataset.
            transform: Optional transform to be applied to each image.
            download: Whether to download the dataset if not found at root.
        """
        self.transform = transform
        self.dataset = Caltech101(root=root, transform=None, download=download)
        self.y = self.dataset.y
        self.categories = self.dataset.categories

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        """
        Retrieve an image and label at the given index. Ensures the image is in RGB mode.

        Returns:
            A tuple containing the transformed image and its label.
        """
        image, label = self.dataset[index]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
