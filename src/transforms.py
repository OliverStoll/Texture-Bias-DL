from common_utils.logger import create_logger
import albumentations as A

from eval_transforms.grid_shuffle import PatchShuffleTransform
from eval_transforms.sobel import SobelFilterTransform
from eval_transforms.canny import CannyFilterTransform
from eval_transforms.bilateral import BilateralFilterTransform
from eval_transforms.median import MedianFilterTransform
from eval_transforms.gaussian_blur import GaussianBlurTransform
from eval_transforms.channel_shuffle import ChannelShuffleTransform
from eval_transforms.channel_inversion import ChannelInversionTransform
from eval_transforms.greyscale import GrayScaleTransform
from eval_transforms.color_jitter import ColorJitterTransform
from eval_transforms.patch_rotation import PatchRotationTransform
from sanity_checks.check_transforms import test_transform
from datasets import DataLoaderFactory



empty_transforms = [{
    'type': None,
    'param': None,
    'param_name': None,
    'transform': None,
}]



class TransformFactory:
    log = create_logger("TransformFactory")

    def __init__(self):
        self.transforms = {
            'bilateral': {
                'class': BilateralFilterTransform,
                'param_name': 'd',
                'param_values': [0, 3, 5, 9, 15, 31],
                'param_values_minimal': [5, 23],
            },
            'median': {
                'class': MedianFilterTransform,
                'param_name': 'kernel_size',
                'param_values': [0, 3, 5, 9, 15, 31],
                'param_values_minimal': [3, 5],
            },
            'gaussian': {
                'class': GaussianBlurTransform,
                'param_name': 'sigma',
                'param_values': [0., 0.5, 1., 2., 3., 5., 9., 15.],
                'param_values_minimal': [1., 5.],
            },
            'sobel': {
                'class': SobelFilterTransform,
                'param_name': 'kernel_size',
                'param_values': [0, 3],  # TODO: implement more kernel sizes
                'param_values_minimal': [0, 3],
            },
            # 'canny': {
            #     'class': CannyFilterTransform,
            #     'param_name': 'thresholds',
            #     'param_values_minimal': [(50, 150), (100, 200), (150, 250)],
            # },
            'patch_shuffle': {
                'class': PatchShuffleTransform,
                'param_name': 'grid_size',
                'param_values': [1, 2, 4, 6, 8, 11, 15],
                'param_values_minimal': [1, 4, 8, 15],
            },
            'patch_rotation': {
                'class': PatchRotationTransform,
                'param_name': 'grid_size',
                'param_values': [1, 2, 4, 6, 8, 11, 15],
                'param_values_minimal': [1, 4, 8, 15],
            },
            'channel_shuffle': {
                'class': ChannelShuffleTransform,
                'param_name': 'n',
                'param_values': [0, 2, 3, 6, 12],
                'param_values_minimal': [0, 2, 12],
            },
            'channel_inversion': {
                'class': ChannelInversionTransform,
                'param_name': 'n',
                'param_values': [0, 1, 2, 3, 6, 12],
                'param_values_minimal': [0, 1, 2, 12]
            },
            'greyscale': {
                'class': GrayScaleTransform,
                'param_name': 'enabled',
                'param_values': [0, 1],
                'param_values_minimal': [0, 1],
            }
        }


    def get_transforms(self, transform_name, param_name=None, param_values=None):
        # todo: add multi-parameter support
        assert not (param_name is not None and param_values is None), "If param_name is provided, param_values must be provided as well."
        if param_name is None:
            param_name = self.transforms[transform_name]['param_name']
        if param_values is None:
            param_values = self.transforms[transform_name]['param_values']
        elif not isinstance(param_values, list):
            param_values = [param_values]
        transform_dicts = [
            {
                'type': transform_name,
                'param': param,
                'param_name': param_name,
                'transform': self.transforms[transform_name]['class'](**{param_name: param})
            } for param in param_values
        ]
        # self.log.info(f"Created {len(transform_dicts)} transforms of type {transform_name} with parameter {param_values}.")
        return transform_dicts

    def get_default_transform(self, transform_name, use_minimal=False):
        transform = self.transforms[transform_name]
        params = transform['param_values_minimal'] if use_minimal else transform['param_values']
        return self.get_transforms(transform_name, transform['param_name'], params)

    def get_all_default_transforms(self, use_minimal=False):
        all_transforms = []
        for transform_name, transform in self.transforms.items():
            params = transform['param_values_minimal'] if use_minimal else transform['param_values']
            transforms = self.get_transforms(transform_name, transform['param_name'], params)
            all_transforms.extend(transforms)
        return all_transforms


if __name__ == '__main__':
    for transform in TransformFactory().get_all_default_transforms():
        for dataset in DataLoaderFactory().dataset_names:
            test_transform(
                transform=transform['transform'],
                transform_name=transform['type'],
                param=transform['param'],
                dataset=dataset,
            )
            try:
                pass
            except Exception as e:
                print(f"Error in transform {transform['type']} on dataset {dataset}.")