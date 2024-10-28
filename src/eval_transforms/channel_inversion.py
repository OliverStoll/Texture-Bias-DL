import torch
import random

from sanity_checks.check_transforms import test_transform


class ChannelInversionTransform(object):
    def __init__(self, n):
        """
        Args:
            n (int, optional): Number of channels to invert.
        """
        self.num_channels_to_invert = n

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor with shape (C, H, W), where C is the number of channels.
        Returns:
            Tensor: Image with inverted channels.
        """

        channels = img.shape[0]
        if self.num_channels_to_invert == 0:
            return img

        # Determine how many channels to shuffle
        if self.num_channels_to_invert > channels:
            num_channels_to_invert = channels
        else:
            num_channels_to_invert = self.num_channels_to_invert

        # Get the indices of channels to shuffle
        channel_indices = list(range(channels))
        channels_to_invert = random.sample(channel_indices, num_channels_to_invert)

        # invert by multiplying by -1
        img[channels_to_invert] = -1 * img[channels_to_invert]

        return img


if __name__ == "__main__":
    for param in [0, 1, 2, 3, 7, 12]:
        for dataset in ['imagenet', 'bigearthnet']:
            test_transform(
                ChannelInversionTransform(n=param),
                "channel_inversion",
                param,
                dataset
            )
