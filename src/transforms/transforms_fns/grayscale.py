import torch
import torchvision.transforms as T

from sanity_checks.check_transforms import test_transform


class GrayScaleTransform:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts a multi-channel image to grayscale and replicates the grayscale
        values across the original number of channels.

        Args:
            image (torch.Tensor): Input image tensor of shape [C, H, W].

        Returns:
            torch.Tensor: Grayscale image replicated across the original number of channels.
        """
        if self.enabled == 0:
            return image

        # Compute the grayscale image by averaging over the channels
        grayscale = torch.mean(image, dim=0, keepdim=True)  # Shape: [1, H, W]

        # Duplicate the grayscale channel to match the original number of channels
        grayscale_replicated = grayscale.repeat(image.shape[0], 1, 1)  # Shape: [C, H, W]

        return grayscale_replicated


if __name__ == "__main__":
    # Define the transform
    transform = GrayScaleTransform()

    # Run the sanity check
    for dataset in ['bigearthnet', 'imagenet']:
        test_transform(
            transform,
            transform_name='grayscale',
            param='',
            dataset=dataset
        )
