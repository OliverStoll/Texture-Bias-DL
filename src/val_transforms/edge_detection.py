import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


from util_code.example_image import get_example_image_tensor, to_tensor, to_image


class EdgeDetectionTransform:
    def __init__(self, transform_type='sobel', output_image=False):
        self.output_image = output_image
        self.transform_type = transform_type
        match transform_type:
            case 'sobel':
                self.filter_function = self.sobel
            case 'canny':
                self.filter_function = self.canny
            case _:
                raise ValueError(f"Unknown transform type: {transform_type}")

    def __call__(self, image_tensor):
        return self.filter_function(image_tensor)

    def sobel(self, image_tensor):
        """Applies a Sobel edge detection filter to an image, channel-wise"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Reshape the kernels to be applied as convolution filters
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        # Convert image to tensor
        img_tensor = image_tensor.unsqueeze(0)
        channels = img_tensor.size(1)
        edges_tensor = torch.zeros_like(img_tensor)

        # Apply Sobel filter to each channel individually
        for c in range(channels):
            channel_img = img_tensor[:, c:c+1, :, :]
            edge_x = F.conv2d(channel_img, sobel_x, padding=1)
            edge_y = F.conv2d(channel_img, sobel_y, padding=1)
            edges_tensor[:, c:c+1, :, :] = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edges_tensor = edges_tensor.squeeze(0)  # Shape: (C, H, W)
        edges_output = transforms.ToPILImage()(edges_tensor) if self.output_image else edges_tensor
        return edges_output

    def canny(self, image_tensor, low_threshold=100, high_threshold=200):
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
        edges_output = transforms.ToPILImage()(edges_tensor) if self.output_image else edges_tensor
        return edges_output


if __name__ == '__main__':
    tensor = get_example_image_tensor()
    trans_type = 'canny'
    edge_detection_transform = EdgeDetectionTransform(transform_type=trans_type, output_image=True)
    edges_image = edge_detection_transform(tensor)
    # save
    edges_image.save("edges.png")
    image = to_image(tensor)
    image.save("original_image.png")