import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift


class HighPassFilterTransform:
    def __init__(self, cutoff: float):
        """
        Args:
            cutoff: The cutoff frequency for the high-pass filter.
        """
        self.cutoff_frequency = cutoff

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply high-pass filter to the input image.

        Args:
            img: Input image tensor of shape (C, H, W) where C is the number of channels.

        Returns:
            Tensor: Filtered image.
        """
        # Ensure the input is a 2D grayscale or multi-channel image
        if img.dim() == 2:
            img = img.unsqueeze(0)  # Add channel dimension for grayscale images

        # Apply the FFT to transform to frequency domain
        img_fft = fft2(img, dim=(-2, -1))
        img_fft_shifted = fftshift(img_fft)

        # Create the high-pass filter mask
        _, H, W = img.size()
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        center = torch.tensor([H // 2, W // 2])
        dist_from_center = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        high_pass_mask = (dist_from_center > self.cutoff_frequency).float()

        # Apply the mask in the frequency domain
        img_fft_shifted = img_fft_shifted * high_pass_mask

        # Inverse FFT to transform back to the spatial domain
        img_filtered = ifftshift(img_fft_shifted)
        img_filtered = ifft2(img_filtered, dim=(-2, -1))

        # Return the real part of the inverse FFT
        return img_filtered.real
