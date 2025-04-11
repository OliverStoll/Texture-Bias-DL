import torch

from tests.transforms import test_transform


class NoiseFilterTransform:
    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.intensity == 0:
            return img

        noise = torch.randn_like(img) * self.intensity
        remaining_img = img * (1 - self.intensity)
        noisy_img = remaining_img + noise
        return noisy_img


if __name__ == "__main__":
    for intensity in [0.1, 0.5, 1.0]:
        test_transform(
            transform=NoiseFilterTransform(intensity=intensity),
            transform_name="noise",
            param=intensity,
            dataset='deepglobe'
        )