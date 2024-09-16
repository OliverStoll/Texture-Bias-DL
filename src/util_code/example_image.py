import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

from utils.config import CONFIG


def get_example_image():
    return cv2.imread(CONFIG['example_image'])


def get_example_image_tensor():
    image = get_example_image()
    tensor = transforms.ToTensor()(image)
    return tensor


to_tensor = ToTensor()
to_image = ToPILImage()