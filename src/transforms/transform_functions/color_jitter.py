import torch
import random
from torchvision.transforms import ColorJitter
import numpy as np

from checks.transforms import test_transform


class ColorJitterTransform:
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def adjust_brightness(self, img, factor):
        # Adjust brightness by adding a factor, not multiplying
        return img + factor

    def adjust_contrast(self, img, factor):
        mean = torch.mean(img, dim=(1, 2), keepdim=True)
        return (img - mean) * factor + mean

    def adjust_saturation(self, img, factor):
        gray = torch.mean(img, dim=0, keepdim=True)
        return (img - gray) * factor + gray

    def __call__(self, tensor_image):
        if self.brightness:
            brightness_factor = random.uniform(-self.brightness, self.brightness)
            tensor_image = self.adjust_brightness(tensor_image, brightness_factor)

        if self.contrast:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            tensor_image = self.adjust_contrast(tensor_image, contrast_factor)

        if self.saturation:
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            tensor_image = self.adjust_saturation(tensor_image, saturation_factor)

        return tensor_image


if __name__  == '__main__':
    for dataset in ['bigearthnet', 'imagenet']:
        for param in [0.5]:
            test_transform(
                ColorJitterTransform(brightness=param),
                "color_jitter",
                param,
                dataset
            )