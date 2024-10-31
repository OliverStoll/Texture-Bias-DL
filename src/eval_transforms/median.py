import torch
import albumentations as A

from sanity_checks.check_transforms import test_transform
from util_code.image_normalization import convert_tensor_to_np_uint8, convert_np_uint8_to_tensor


class MedianFilterTransform:
    def __init__(self, kernel_size: int = 5):
        """
        Initializes the MedianBlurTransform with a given kernel size.
        :param kernel_size: The size of the kernel to be used for the median blur. Should be an odd number.
        """
        self.kernel_size = kernel_size
        if kernel_size != 0:
            self.transform = A.MedianBlur(blur_limit=(kernel_size, kernel_size), p=1)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 0:
            return img
        uint8_image = convert_tensor_to_np_uint8(img)
        filtered_uint8_image = self.transform(image=uint8_image)['image']
        filtered_tensor = convert_np_uint8_to_tensor(filtered_uint8_image)

        return filtered_tensor


if __name__ == "__main__":
    for kernel_size in [7, 9, 15]:
        test_transform(
            transform=MedianFilterTransform(kernel_size=kernel_size),
            transform_name="median",
            param=kernel_size,
            dataset='deepglobe'
        )