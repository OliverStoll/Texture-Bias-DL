import torch
import os
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from common_utils.config import CONFIG

from transforms.transforms_factory import TransformFactory



to_tensor = ToTensor()
to_image = ToPILImage()

BEN_RGB_CHANNELS = (3, 2, 1)
ben_mean = {
    "B01": 361.0660705566406,
    "B02": 438.385009765625,
    "B03": 614.0690307617188,
    "B04": 588.4198608398438,
    "B05": 942.854248046875,
    "B06": 1769.951904296875,
    "B07": 2049.574951171875,
    "B08": 2193.320556640625,
    "B09": 2241.467529296875,
    "B11": 1568.229736328125,
    "B12": 997.7351684570312,
    "B8A": 2235.580810546875,
}
ben_std = {
    "B01": 575.0419311523438,
    "B02": 607.0984497070312,
    "B03": 603.3590087890625,
    "B04": 684.6243896484375,
    "B05": 738.4649047851562,
    "B06": 1100.47900390625,
    "B07": 1275.8310546875,
    "B08": 1369.4029541015625,
    "B09": 1316.40380859375,
    "B11": 1070.170166015625,
    "B12": 813.5411376953125,
    "B8A": 1356.568359375,
}
UINT16_MAX = 65535
ben_means = [ben_mean[key] for key in sorted(ben_mean.keys())]
ben_stds = [ben_std[key] for key in sorted(ben_std.keys())]
ben_means_for_img = [mean / UINT16_MAX for mean in ben_means]
ben_stds_for_img = [std / UINT16_MAX for std in ben_stds]
rgb_ben_means_for_img = [ben_means_for_img[i] for i in BEN_RGB_CHANNELS]
rgb_ben_stds_for_img = [ben_stds_for_img[i] for i in BEN_RGB_CHANNELS]


MEANS_TO_ZERO_ONE = {
    'bigearthnet': ben_means_for_img,
    'rgb_bigearthnet': rgb_ben_means_for_img,
    'imagenet': [0.485, 0.456, 0.406],
    'caltech': [0.485, 0.456, 0.406],
    'caltech_120': [0.485, 0.456, 0.406],
    'deepglobe': [0.4095, 0.3808, 0.2836],
}
STDS_TO_ZERO_ONE = {
    'bigearthnet': ben_means_for_img,
    'rgb_bigearthnet': rgb_ben_stds_for_img,
    'imagenet': [0.229, 0.224, 0.225],
    'caltech': [0.229, 0.224, 0.225],
    'caltech_120': [0.229, 0.224, 0.225],
    'deepglobe': [0.1509, 0.1187, 0.1081],
}


def normalize_ben_rgb(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a BEN RGB tensor of shape (3, H, W) in the order (B04, B03, B02).
    Returns the normalized tensor (channel-wise standardization).
    """
    # Extract channel-wise means and stds for B04, B03, B02
    means = [ben_mean["B04"], ben_mean["B03"], ben_mean["B02"]]
    stds = [ben_std["B04"], ben_std["B03"], ben_std["B02"]]

    # Perform in-place normalization: (value - mean) / std
    for i in range(len(BEN_RGB_CHANNELS)):
        rgb_tensor[i] = (rgb_tensor[i] - means[i]) / stds[i]

    return rgb_tensor


def get_example_tensor(dataset, example_idx=0):
    tensor_path = f"{CONFIG['work_dir']}/{CONFIG['example_data_path']}/{dataset}_{example_idx}.pt"
    tensor = torch.load(tensor_path)
    return tensor


def _clamp_99_percentile(tensor_channel):
    """ Clamp the 99 percentile of the tensor to 1.0 """
    min_val = torch.quantile(tensor_channel, 0.01)
    max_val = torch.quantile(tensor_channel, 0.99)
    tensor_channel[tensor_channel < min_val] = min_val
    tensor_channel[tensor_channel > max_val] = max_val
    tensor_channel = (tensor_channel - min_val) / (max_val - min_val)
    return tensor_channel


def denormalize_tensor_to_ZERO_ONE(tensor, dataset):
    means = MEANS_TO_ZERO_ONE[dataset]
    stds = STDS_TO_ZERO_ONE[dataset]
    new_channels = []
    for tensor_channel, mean, std in zip(tensor, means, stds):
        new_channel = tensor_channel * std + mean
        if 'bigearthnet' in dataset:
            new_channel = _clamp_99_percentile(new_channel)
        new_channels.append(new_channel)
        new_channel[new_channel < 0] = 0
    new_tensor = torch.stack(new_channels)
    return new_tensor


def test_tensor_is_normalized(tensor, max_absolute_value=20):
    """ Check if any value is greater than +-20"""
    return torch.all(torch.abs(tensor) < max_absolute_value)


def test_transform(transform, transform_name, param, dataset, example_idx=0):
    save_path = f"{CONFIG['work_dir']}/{CONFIG['example_image_output']}/{dataset}"
    os.makedirs(save_path, exist_ok=True)
    original_tensor = get_example_tensor(dataset, example_idx=example_idx)
    if not test_tensor_is_normalized(original_tensor) and dataset == 'bigearthnet':
        original_tensor = normalize_ben_rgb(original_tensor)

    transformed_tensor = original_tensor.clone()
    assert test_tensor_is_normalized(original_tensor), "Original tensor is not normalized."
    transformed_tensor = transform(transformed_tensor)
    transformed_tensor = denormalize_tensor_to_ZERO_ONE(transformed_tensor, dataset)
    original_tensor = denormalize_tensor_to_ZERO_ONE(original_tensor, dataset)
    if torch.any(transformed_tensor < 0) or torch.any(transformed_tensor > 1):
        # print(f"Tensor values are not in [0, 1].")
        print(f"Tensor values are not in [0, 1] - "
              f"Min: {torch.min(transformed_tensor):.2f}, Max: {torch.max(transformed_tensor):.2f}")
        transformed_tensor[transformed_tensor < 0] = 0
        transformed_tensor[transformed_tensor > 1] = 1


    transformed_image = to_image(transformed_tensor)
    example_img_path = f"{save_path}/{transform_name}_{param}.png"
    transformed_image.save(example_img_path)

    original_image = to_image(original_tensor)
    if param == 0 and not torch.allclose(original_tensor, transformed_tensor, atol=1e-5):
        original_img_path = f"{save_path}/{dataset}_original.png"
        original_image.save(original_img_path)
        print(f"Transform {transform_name} changed image although param = 0.")


def show_original():
    save_path = f"{CONFIG['work_dir']}/{CONFIG['example_image_output']}/original"
    os.makedirs(save_path, exist_ok=True)
    base_tensor = get_example_tensor('bigearthnet', example_idx=1)
    # get all pair combinations for indexes between 0 and 11:
    possible_channels = [1, 2, 3, 4]
    for i in possible_channels:
        for j in possible_channels:
            for k in possible_channels:
                if i == j or i == k or j == k:
                    continue
                original_tensor = base_tensor.clone()
                channel_triplet = (i, j, k)
                original_tensor = original_tensor[channel_triplet, :, :]
                original_tensor = denormalize_tensor_to_ZERO_ONE(original_tensor, 'bigearthnet')
                original_image = to_image(original_tensor)
                original_img_path = f"{save_path}/bigearthnet_{i}_{j}_{k}.png"
                original_image.save(original_img_path)


def test_all_transforms(transform_factory, dataloader_factory, transform_name=None, datasets=None, example_idx=0):
    if transform_name is None:
        transforms = transform_factory().get_all_default_transforms()
    else:
        transforms = transform_factory().get_multiple_transforms(transform_name)
    datasets = dataloader_factory().dataset_names if datasets is None else datasets
    for transform in transforms:
        for dataset in datasets:
            test_transform(
                transform=transform['transform'],
                transform_name=transform['type'],
                param=transform['param'],
                dataset=dataset,
                example_idx=example_idx,
            )

def manual_test_transforms():
    """ Test script for verifying transform instantiation via TransformFactory. """
    img_idx = {'imagenet': 1, 'deepglobe': 1}
    dataset_names = ['imagenet', 'bigearthnet', 'caltech', 'deepglobe']
    for dataset_name in dataset_names:
        example_idx = img_idx.get(dataset_name, 0)

        print(f"Plotting Single {dataset_name}")
        for single_transform in TransformFactory().get_all_default_transforms():
            test_transform(
                transform=single_transform['transform'],
                transform_name=single_transform['type'],
                param=single_transform['param'],
                dataset=dataset_name,
                example_idx=example_idx
            )

        print(f"Plotting Double {dataset_name}")
        for single_pair in TransformFactory().get_pair_combinations_of_default_transforms():
            test_transform(
                transform=single_pair['transform'],
                transform_name=single_pair['type'],
                param=single_pair['param'],
                dataset=dataset_name,
                example_idx=example_idx
            )


if __name__ == '__main__':
        test_all_transforms(
        example_idx=3,
        transform_name='channel_inversion',
        datasets=['caltech'],
    )


