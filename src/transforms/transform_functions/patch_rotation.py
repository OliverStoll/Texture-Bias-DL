import cv2
import numpy as np
import torch
import random
import torchvision.transforms.functional as torchfunc
import torch.nn.functional as F

from tests.transforms import test_transform


class PatchRotationTransform:
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
        """Takes an image, splits it into grid_size x grid_size patches and rotate them"""
        if self.grid_size == 0:
            return image

        image, dims = self.resize_to_fit_grid(image)
        channels, rows, cols = image.shape
        patch_width = cols // self.grid_size
        patch_height = rows // self.grid_size
        grid_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        rotated_image = torch.zeros_like(image)

        # Iterate over the patch positions and copy the corresponding patch from the original image
        for i, (patch_row, patch_col) in enumerate(grid_positions):
            k = random.randint(1, 3)
            start_row = patch_row * patch_height
            end_row = start_row + patch_height
            start_col = patch_col * patch_width
            end_col = start_col + patch_width
            area = image[
                :,
                (i // self.grid_size) * patch_height: (i // self.grid_size + 1) * patch_height,
                (i % self.grid_size) * patch_width: (i % self.grid_size + 1) * patch_width
            ]
            rotated_image[:, start_row:end_row, start_col:end_col] = torch.rot90(
                input=area, k=k, dims=(1, 2)
            )

        rescaled_rotated_image = self.rescale_to_original(rotated_image)

        return rescaled_rotated_image


if __name__ == '__main__':
    dataset = 'caltech'
    for grid_size in [0, 2, 4, 6, 8, 11, 15]:
        transform = PatchRotationTransform(grid_size=grid_size)
        test_transform(transform, transform_name='patch_rotation', dataset=dataset, param=grid_size)