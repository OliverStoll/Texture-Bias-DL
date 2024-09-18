import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_lightning import LightningModule

from util_code.example_image import test_transform


class MedianFilterTransform:
    def __init__(self, kernel_size: int = 5):
        """
        Initializes the MedianBlurTransform with a given kernel size.
        :param kernel_size: The size of the kernel to be used for the median blur. Should be an odd number.
        """
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply median blur to a given image tensor.
        :param img: The input image tensor in the shape [C, H, W] with values in the range [0, 1].
        :return: A tensor with the same shape but with median filtering applied.
        """

        # Convert the tensor from [C, H, W] to [H, W, C] and scale to [0, 255] for OpenCV compatibility
        img = img.permute(1, 2, 0).numpy() * 255.0  # Convert from [C, H, W] to [H, W, C] and scale to [0, 255]
        img = img.astype(np.uint8)  # Convert to uint8 data type

        # Apply median blur using OpenCV
        img = cv2.medianBlur(img, self.kernel_size)

        # Convert back to a PyTorch tensor and re-scale to [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # Convert back to [C, H, W]

        return img


if __name__ == "__main__":
    test_transform(
        transform=MedianFilterTransform(kernel_size=5),
    )