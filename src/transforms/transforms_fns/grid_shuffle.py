import cv2
import numpy as np
import torch
import torchvision.transforms.functional as torchfunc
import torch.nn.functional as F

from sanity_checks.check_transforms import test_transform


class PatchShuffleTransform:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.dims = None

    def resize_to_fit_grid(self, tensor):
        channels, rows, cols = tensor.shape
        self.dims = (rows, cols)
        new_rows = (rows // self.grid_size) * self.grid_size
        new_cols = (cols // self.grid_size) * self.grid_size
        resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(new_rows, new_cols), mode='nearest')
        return resized_tensor.squeeze(0), (rows, cols)

    def rescale_to_original(self, tensor):
        """Rescale the shuffled tensor back to the original dimensions"""
        original_sized_tensor = F.interpolate(tensor.unsqueeze(0), size=self.dims, mode='nearest')
        return original_sized_tensor.squeeze(0)


    def __call__(self, image):
        """Takes an image, splits it into grid_size x grid_size patches and shuffles them"""
        if self.grid_size == 0:
            return image
        image, dims = self.resize_to_fit_grid(image)
        channels, rows, cols = image.shape
        patch_width = cols // self.grid_size
        patch_height = rows // self.grid_size
        grid_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        np.random.shuffle(grid_positions)  # noqa
        shuffled_image = torch.zeros_like(image)

        # Iterate over the grid positions and copy the corresponding grid from the original image
        for i, (grid_row, grid_col) in enumerate(grid_positions):
            start_row = grid_row * patch_height
            end_row = start_row + patch_height
            start_col = grid_col * patch_width
            end_col = start_col + patch_width
            shuffled_image[:, start_row:end_row, start_col:end_col] = image[
                :,
                (i // self.grid_size) * patch_height: (i // self.grid_size + 1) * patch_height,
                (i % self.grid_size) * patch_width: (i % self.grid_size + 1) * patch_width
            ]

        rescaled_shuffled_image = self.rescale_to_original(shuffled_image)

        return rescaled_shuffled_image


if __name__ == '__main__':
    transform = GridShuffleTransform(4)
    test_transform(transform, 'grid_shuffle', 5, dataset='bigearthnet')