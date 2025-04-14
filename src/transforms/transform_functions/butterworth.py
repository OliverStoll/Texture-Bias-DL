import torch
import torch.fft

from checks.transforms import get_example_tensor, to_tensor, to_image


class ButterworthHighPassFilter:
    # TODO: FIX (BROKEN)
    def __init__(self, cutoff=30, order=2):
        """
        Initialize the ButterworthHighPassFilter.

        Args:
            cutoff (float): Cutoff frequency in pixels.
            order (int): Order of the filter (higher order means a sharper cutoff).
        """
        self.cutoff = cutoff
        self.order = order

    def __call__(self, image_tensor):
        """
        Apply Butterworth high-pass filter to the input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W).

        Returns:
            torch.Tensor: Filtered image tensor with the same shape as input.
        """
        # Ensure the image tensor has 4 dimensions (N, C, H, W)
        original_shape = image_tensor.shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() != 4:
            raise ValueError("image_tensor must be a 3D or 4D tensor.")

        N, C, H, W = image_tensor.shape

        # Move to frequency domain
        # Shift the zero-frequency component to the center of the spectrum
        fft_image = torch.fft.fftshift(torch.fft.fft2(image_tensor, dim=(-2, -1)), dim=(-2, -1))

        # Create the Butterworth high-pass filter
        u = torch.arange(-W // 2, W // 2, device=image_tensor.device).view(1, 1, 1, W)
        v = torch.arange(-H // 2, H // 2, device=image_tensor.device).view(1, 1, H, 1)
        D_uv = torch.sqrt(u**2 + v**2)
        H_uv = 1 / (1 + (self.cutoff / (D_uv + 1e-8)) ** (2 * self.order))

        # Expand H_uv to match the batch and channel dimensions
        H_uv = H_uv.expand(N, C, H, W)

        # Apply the filter in frequency domain
        fft_filtered = fft_image * H_uv

        # Shift back and perform inverse FFT
        fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        image_filtered = torch.fft.ifft2(fft_filtered, dim=(-2, -1)).real

        # Ensure the output has the same shape as the input
        if len(original_shape) == 3:
            image_filtered = image_filtered.squeeze(0)

        return image_filtered


if __name__ == '__main__':
    tensor = get_example_tensor()
    transform = ButterworthHighPassFilter(cutoff=50, order=5)
    output_tensor = transform(tensor)
    output_image = to_image(output_tensor)
    # save
    output_image.save("output.png")
    image = to_image(tensor)
    image.save("image.png")