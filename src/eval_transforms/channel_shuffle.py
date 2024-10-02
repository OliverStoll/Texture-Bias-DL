import torch
import random

from sanity_checks.check_transforms import test_transform


class ChannelShuffleTransform(object):
    def __init__(self, n):
        """
        Args:
            n (int, optional): Number of channels to shuffle.
                                                     If None, all channels are shuffled.
        """
        self.num_channels_to_shuffle = n

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor with shape (C, H, W), where C is the number of channels.
        Returns:
            Tensor: Image with shuffled channels.
        """

        channels = img.shape[0]
        if self.num_channels_to_shuffle == 0:
            return img

        # Determine how many channels to shuffle
        if self.num_channels_to_shuffle > channels:
            num_channels_to_shuffle = channels
        else:
            num_channels_to_shuffle = self.num_channels_to_shuffle

        # Get the indices of channels to shuffle
        channel_indices = list(range(channels))
        channels_to_shuffle = random.sample(channel_indices, num_channels_to_shuffle)

        # Ensure the shuffle is different from the original order
        shuffled_indices = channels_to_shuffle.copy()
        while shuffled_indices == channels_to_shuffle:
            random.shuffle(shuffled_indices)

        # Now we reorder the entire channel list
        new_order = channel_indices[:]
        for i, shuffled in enumerate(shuffled_indices):
            new_order[channels_to_shuffle[i]] = shuffled

        # Reorder the image channels
        img = img[new_order, :, :]

        return img


if __name__ == "__main__":
    for param in [0, 2, 3, 4, 5, None]:
        for dataset in ['imagenet', 'bigearthnet']:
            test_transform(
                ChannelShuffleTransform(n=param),
                "channel_shuffle",
                param,
                dataset
            )
