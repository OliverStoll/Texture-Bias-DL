import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

from tests.transforms import test_transform


class CannyFilterTransform:
    def __init__(self, thresholds=(100, 230)):
        self.low_threshold, self.high_threshold = thresholds

    def __call__(self, image_tensor):
        if image_tensor.ndimension() != 3:
            raise ValueError("Input tensor should have 3 dimensions (C, H, W)")

        img_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img_np = img_np.astype(np.uint8)
        # convert to grayscale, without cv

        edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold)
        #  add channel dimensions
        edges = np.expand_dims(edges, axis=2)
        edges_tensor = torch.tensor(edges).permute(2, 0, 1).float() / 255.0
        # invert the image
        edges_tensor = 1 - edges_tensor
        return edges_tensor


if __name__ == '__main__':
    test_transform(CannyFilterTransform((100, 230)))