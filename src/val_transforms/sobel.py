import torch
import torch.nn.functional as F

from util_code.example_image import test_transform


class SobelFilterTransform:
    def __init__(self):
        pass

    def __call__(self, image_tensor):
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
        return edges_tensor


if __name__ == '__main__':
    test_transform(SobelFilterTransform())