import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor

from data_loading.dataloader import get_dataloader


def grid_shuffle(image, grid_size, resize_to=None):
    # resize the image to 256x256
    if resize_to is not None:
        image = cv2.resize(image, resize_to)

    # Divide the image into a 4x4 grid
    rows, cols, _ = image.shape
    grid_width = cols // grid_size
    grid_height = rows // grid_size

    # Create a list of grid positions
    grid_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    # Shuffle the grid positions randomly
    np.random.shuffle(grid_positions)

    # Create a new image to store the shuffled grid
    shuffled_image = np.zeros_like(image)

    # Iterate over the grid positions and copy the corresponding grid from the original image to the shuffled image
    for i, (grid_row, grid_col) in enumerate(grid_positions):
        # Calculate the coordinates of the current grid in the original image
        start_row = grid_row * grid_height
        end_row = start_row + grid_height
        start_col = grid_col * grid_width
        end_col = start_col + grid_width

        # Copy the grid from the original image to the shuffled image
        shuffled_image[start_row:end_row, start_col:end_col] = image[
            i // grid_size * grid_height : (i // grid_size + 1) * grid_height,
            i % grid_size * grid_width : (i % grid_size + 1) * grid_width,
        ]

    return shuffled_image


def make_imagenet_tests():
    import os
    root_folder = "/media/storagecube/data/shared/datasets/ImageNet-2012"

    # Get list of directories in the root folder
    dirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    # Sort the directories and take the first one
    for i in range(1, 1001):
        first_dir = sorted(dirs)[i]
        first_dir_path = os.path.join(root_folder, first_dir)

        # Get list of files in the first directory
        files = [f for f in os.listdir(first_dir_path)]

        # Sort the files and take the first one
        first_file = sorted(files)[0]
        first_file_path = os.path.join(first_dir_path, first_file)

        image = cv2.imread(first_file_path)

        # Apply grid shuffle
        shuffled_image = grid_shuffle(image, grid_size=4)

        # save the shuffled image
        cv2.imwrite(f"/home/olivers/colab-master-thesis/_imagetests/{i}.jpg", shuffled_image)
        cv2.imwrite(f"/home/olivers/colab-master-thesis/_imagetests/{i}_original.jpg", image)


def make_bigearthnet_tests():
    _, loader, _ = get_dataloader('bigearthnet')
    # get image with iter
    iterator = iter(loader)
    images, _ = next(iterator)

    # Apply grid shuffle
    image = images[0].numpy().transpose(1, 2, 0)
    shuffled_image = grid_shuffle(image, grid_size=4)

    # only save channels 4 3 2
    shuffled_image = shuffled_image[:, :, [3, 2, 1]]

    # Save the shuffled image
    cv2.imwrite("shuffled_image.jpg", shuffled_image)


def read_lmdb_first_image():
    pass


if __name__ == "__main__":
    make_bigearthnet_tests()

