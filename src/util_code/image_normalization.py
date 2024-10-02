import numpy as np
import torch


def convert_tensor_to_np_uint8(image_tensor, clip_std_devs=3):
    permuted_tensor = image_tensor.permute(1, 2, 0)
    image_data = permuted_tensor.cpu().numpy()
    clipped_image = np.clip(image_data, -clip_std_devs, clip_std_devs)
    # Rescale to the 0-255 uint8 range
    rescaled_image = 255 * (clipped_image + clip_std_devs) / (2 * clip_std_devs)
    uint8_image = rescaled_image.astype(np.uint8)
    return uint8_image


def convert_np_uint8_to_tensor(uint8_image, clip_std_devs=3):
    float_image = uint8_image.astype(np.float32)
    normalized_image = (float_image / 255) * (2 * clip_std_devs) - clip_std_devs
    normalized_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1)
    return normalized_tensor



