import cv2
import numpy as np
import torchvision.transforms.functional as torchfunc


class GridShuffleTransform:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __call__(self, image):
        """Takes an image, splits it into grid_size x grid_size patches and shuffles them"""

        # Divide the image into a grid
        _, rows, cols = image.shape
        grid_width = cols // self.grid_size
        grid_height = rows // self.grid_size
        grid_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        np.random.shuffle(grid_positions)  # noqa
        shuffled_image = np.zeros_like(image)

        # Iterate over the grid positions and copy the corresponding grid from the original image
        for i, (grid_row, grid_col) in enumerate(grid_positions):
            start_row = grid_row * grid_height
            end_row = start_row + grid_height
            start_col = grid_col * grid_width
            end_col = start_col + grid_width
            shuffled_image[:, start_row:end_row, start_col:end_col] = image[
                :,
                i // self.grid_size * grid_height: (i // self.grid_size + 1) * grid_height,
                i % self.grid_size * grid_width: (i % self.grid_size + 1) * grid_width
            ]

        shuffled_tensor = torchfunc.to_tensor(shuffled_image)
        shuffled_tensor_perm = shuffled_tensor.permute(1, 2, 0)

        return shuffled_tensor_perm
