import numpy as np
import torch
import cv2
import torchvision.transforms as transforms


class CannyFilterTransform:
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image_tensor):
        if image_tensor.ndimension() != 3:
            raise ValueError("Input tensor should have 3 dimensions (C, H, W)")

        img_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img_np = img_np.astype(np.uint8)
        edges_per_channel = []
        for i in range(img_np.shape[2]):  # Loop over each channel
            channel_img = img_np[:, :, i]  # Extract the i-th channel
            edges = cv2.Canny(channel_img, low_threshold, high_threshold)
            edges_per_channel.append(edges)

        edges_np = np.stack(edges_per_channel, axis=-1)
        edges_tensor = torch.tensor(edges_np).permute(2, 0, 1).float() / 255.0
        return edges_tensor