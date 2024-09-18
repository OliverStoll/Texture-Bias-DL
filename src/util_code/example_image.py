import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

from utils.config import CONFIG


to_tensor = ToTensor()
to_image = ToPILImage()


def get_example_image():
    return cv2.imread(CONFIG['example_image'])


def get_example_image_tensor():
    image = get_example_image()
    tensor = transforms.ToTensor()(image)
    return tensor


def test_transform(transform):
    image_tensor = get_example_image_tensor()
    filtered_tensor = transform(image_tensor)
    image = to_image(filtered_tensor)
    original_image = to_image(image_tensor)
    image.save("transformed_image.png")
    original_image.save("original_image.png")


