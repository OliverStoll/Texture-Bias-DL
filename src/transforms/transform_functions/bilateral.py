import torch
import cv2
import pytorch_lightning as pl
import numpy as np
from scipy.ndimage import gaussian_filter

# import example tensor
from checks.transforms import test_transform
from transforms.transform_functions._image_normalization import convert_tensor_to_np_uint8, convert_np_uint8_to_tensor


class BilateralFilterTransform:
    def __init__(self, d=5, sigma_color=75, sigma_space=75):
        """
        Initializes the bilateral filter parameters.

        Args:
            d (int): Diameter of the pixel neighborhood.
            sigma_color (float): Filter sigma in the color space.
            sigma_space (float): Filter sigma in the coordinate space.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, image_tensor):
        """
        Applies a bilateral filter to an image tensor.

        Args:
            image_tensor (torch.Tensor): A 3D tensor representing the image (C, H, W).
                                         It must be in the range [0, 1] and of type float.

        Returns:
            torch.Tensor: The filtered image tensor.
        """
        if self.d == 0:
            return image_tensor

        assert len(image_tensor.shape) == 3, "The input tensor must be in the format [C, H, W]."

        uint8_numpy_image = convert_tensor_to_np_uint8(image_tensor)
        image_channels = cv2.split(uint8_numpy_image)

        if len(image_channels) != 3:
            filtered_channels = []
            for channel in image_channels:
                filtered_channel = cv2.bilateralFilter(
                    channel, self.d, self.sigma_color, self.sigma_space
                )
                filtered_channels.append(filtered_channel)
            filtered_np_image = cv2.merge(filtered_channels)
        else:
            filtered_np_image = cv2.bilateralFilter(
                uint8_numpy_image, self.d, self.sigma_color, self.sigma_space
            )
        filtered_tensor = convert_np_uint8_to_tensor(filtered_np_image)

        return filtered_tensor


# Example usage
if __name__ == "__main__":
    dataset = 'imagenet'
    for d in [0, 1, 3, 5, 9, 15]:
        bilateral_transform = BilateralFilterTransform(d=d)
        test_transform(
            bilateral_transform,
            "bilateral",
            param=d,
            dataset=dataset,
            example_idx=1,
        )
