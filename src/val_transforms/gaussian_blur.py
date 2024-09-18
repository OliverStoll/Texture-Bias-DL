import torch
import torch.nn.functional as F
import cv2
import pytorch_lightning as pl
import numpy as np

# import example tensor
from util_code.example_image import get_example_image_tensor, to_image


class GaussianBlurTransform:
    def __init__(self, kernel_size=5, sigma=1.0):
        """
        Initialize the GaussianBlurTransform.

        Args:
            kernel_size (int): Size of the Gaussian kernel (must be odd).
            sigma (float): Standard deviation of the Gaussian distribution.
        """
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create a Gaussian kernel
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)

    def __call__(self, image_tensor):
        """
        Apply Gaussian blur to the input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W).

        Returns:
            torch.Tensor: Blurred image tensor with the same shape as input.
        """
        # Ensure the image tensor has 4 dimensions (N, C, H, W)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() != 4:
            raise ValueError("image_tensor must be a 3D or 4D tensor.")

        N, C, H, W = image_tensor.shape

        # Move kernel to the same device as image
        kernel = self.kernel.to(image_tensor.device)

        # Repeat the kernel for each channel
        kernel = kernel.expand(C, 1, self.kernel_size, self.kernel_size)

        # Apply Gaussian blur using depthwise convolution
        blurred = F.conv2d(image_tensor, kernel, groups=C, padding=self.kernel_size // 2)

        # If the input was 3D, remove the batch dimension
        if blurred.shape[0] == 1:
            blurred = blurred.squeeze(0)

        return blurred

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a 2D Gaussian kernel.

        Args:
            kernel_size (int): Size of the kernel.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: 2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
        """
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")

        # Create a coordinate grid
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid([ax, ax], indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)

        # Reshape to match conv2d input
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel


if __name__ == "__main__":
    image_tensor = get_example_image_tensor()
    transform = GaussianBlurTransform()
    filtered_tensor = transform(image_tensor)

    # save the tensor as an image
    image = to_image(filtered_tensor)
    original_image = to_image(image_tensor)
    image.save("gauss_blur_image.png")
    original_image.save("original_image.png")
