import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as torchfunc

from data_init import DataLoaderCollection


class GridShuffleTransform:
    def __init__(self, grid_size, resize_to=None):
        self.grid_size = grid_size
        self.resize_to = resize_to

    def __call__(self, image):
        """Takes an image, splits it into grid_size x grid_size patches and shuffles them"""

        if self.resize_to is not None:
            image = cv2.resize(image, self.resize_to)

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


def visualize_normalized_image(image, dataset):
    image = image.numpy().transpose(1, 2, 0)
    image = image[:, :, [3, 2, 1]] if dataset == 'bigearthnet' else image
    rescaled_image = ((image + 1) / 2) * 255
    return rescaled_image


def test_grid_shuffle(dataset):
    dl_collection = DataLoaderCollection()
    val_transform = GridShuffleTransform(grid_size=4)
    dl_tuple = dl_collection.get_dataloader(dataset_name=dataset, model_name='resnet',
                                            is_pretrained=False, val_transform=val_transform)

    train_dl, val_dl, _ = dl_tuple
    train_iterator = iter(train_dl)
    train_imgs, _ = next(train_iterator)
    val_iterator = iter(val_dl)
    val_imgs, _ = next(val_iterator)

    train_images = train_imgs[:5]
    train_images = [visualize_normalized_image(image, dataset) for image in train_images]
    val_images = val_imgs[:5]
    val_images = [visualize_normalized_image(image, dataset) for image in val_images]

    for i in range(5):
        cv2.imwrite(f"../output/{dataset}/{i}_shuffled.jpg", val_images[i])
        cv2.imwrite(f"../output/{dataset}/{i}_original.jpg", train_images[i])


if __name__ == "__main__":
    test_grid_shuffle(dataset='imagenet')
    test_grid_shuffle(dataset='bigearthnet')

