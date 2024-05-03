import cv2
import numpy as np


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


if __name__ == "__main__":
    # Load an image
    image = cv2.imread("data/dog.jpg")

    # Apply grid shuffle
    shuffled_image = grid_shuffle(image, grid_size=1000, resize_to=(1000, 1000))

    # Display the original and shuffled images
    cv2.imshow("Shuffled Image", shuffled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
