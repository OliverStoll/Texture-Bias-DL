import torch


class LowPassFilterTransform:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, image_tensor):
        # TODO: implement
        pass


if __name__ == "__main__":
    transform = LowPassFilterTransform(filter_type="gaussian", cutoff=20)
    img_tensor = torch.randn((1, 3, 224, 224))  # Example input tensor
    filtered_image = transform(img_tensor)
