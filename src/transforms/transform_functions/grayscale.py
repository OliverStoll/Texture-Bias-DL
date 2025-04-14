import torch
import torchvision.transforms as T

from checks.transforms import test_transform


class GrayScaleTransform:
    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts a multi-channel image to grayscale and replicates the grayscale
        values across the original number of channels.

        Args:
            image (torch.Tensor): Input image tensor of shape [C, H, W].

        Returns:
            torch.Tensor: Grayscale image replicated across the original number of channels.
        """
        if self.percentage == 0:
            return image

        # Compute the grayscale image by averaging over the channels
        averaged = torch.mean(image, dim=0, keepdim=True)  # Shape: [1, H, W]

        # Duplicate the grayscale channel to match the original number of channels
        averaged_all_channels = averaged.repeat(image.shape[0], 1, 1)  # Shape: [C, H, W]

        # multiply the grayscale image by the percentage and add the original image multiplied by 1 - percentage
        averaged_all_channels_by_percentage = averaged_all_channels * self.percentage
        original_image_by_percentage = image * (1 - self.percentage)

        combined_image = averaged_all_channels_by_percentage + original_image_by_percentage

        return combined_image


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
