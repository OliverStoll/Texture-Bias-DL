import torch
import cv2
import pytorch_lightning as pl
import numpy as np

# import example tensor
from util_code.example_image import get_example_image_tensor, to_image


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
        # Check if the input tensor is in the format [C, H, W] and convert it to [H, W, C]
        assert len(image_tensor.shape) == 3, "The input tensor must be in the format [C, H, W]."
        image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Convert the image to the correct type and scale for OpenCV processing (uint8)
        # TODO: fix clipping for all transforms
        image_numpy = np.clip(image_numpy * 255.0, 0, 255).astype(np.uint8)

        # Apply bilateral filter using OpenCV
        filtered_image = cv2.bilateralFilter(image_numpy, self.d, self.sigma_color,
                                             self.sigma_space)

        # Convert the result back to [C, H, W] and normalize it to [0, 1]
        filtered_image = torch.from_numpy(filtered_image.astype(np.float32) / 255.0)
        filtered_image = filtered_image.permute(2, 0, 1)

        return filtered_image


# Example usage
if __name__ == "__main__":
    image_tensor = get_example_image_tensor()
    bilateral_transform = BilateralFilterTransform()
    filtered_tensor = bilateral_transform(image_tensor)

    # save the tensor as an image
    image = to_image(filtered_tensor)
    original_image = to_image(image_tensor)
    image.save("bilateral_filtered_image.png")
    original_image.save("original_image.png")
