from torchvision.transforms import Compose
from common_utils.logger import create_logger

from transforms.transforms_fns.grid_shuffle import PatchShuffleTransform
from transforms.transforms_fns.sobel import SobelFilterTransform
from transforms.transforms_fns.bilateral import BilateralFilterTransform
from transforms.transforms_fns.median import MedianFilterTransform
from transforms.transforms_fns.gaussian_blur import GaussianBlurTransform
from transforms.transforms_fns.channel_shuffle import ChannelShuffleTransform
from transforms.transforms_fns.channel_inversion import ChannelInversionTransform
from transforms.transforms_fns.greyscale import GrayScaleTransform
from transforms.transforms_fns.patch_rotation import PatchRotationTransform

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
                'param_values': [0., 1., 3., 5., 9., 15.],
                'param_values_minimal': [1., 5.],
            },
            # 'sobel': {
            #     'class': SobelFilterTransform,
            #     'param_name': 'kernel_size',
            #     'param_values': [0, 3],  # TODO: implement more kernel sizes
            #     'param_values_minimal': [0, 3],
            # },
            # 'canny': {
            #     'class': CannyFilterTransform,
            #     'param_name': 'thresholds',
            #     'param_values_minimal': [(50, 150), (100, 200), (150, 250)],
            # },
            'patch_shuffle': {
                'class': PatchShuffleTransform,
                'param_name': 'grid_size',
                'param_values': [0, 2, 4, 6, 8, 11, 15],
                'param_values_minimal': [1, 4, 8, 15],
            },
            'patch_rotation': {
                'class': PatchRotationTransform,
                'param_name': 'grid_size',
                'param_values': [0, 2, 4, 6, 8, 11, 15],
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


    def get_multiple_transforms(self, transform_name, param_name=None, param_values=None):
        assert not (param_name is not None and param_values is None), "If param_name is provided, param_values must be provided as well."
        param_name = param_name or self.transforms[transform_name]['param_name']
        param_values = param_values or self.transforms[transform_name]['param_values']
        if not isinstance(param_values, list):
            param_values = [param_values]
        transform_dicts = [
            {
                'type': transform_name,
                'param': param,
                'param_name': param_name,
                'transform': self.transforms[transform_name]['class'](**{param_name: param})
            } for param in param_values
        ]
        self.log.debug(f"Created {len(transform_dicts)} transforms of type {transform_name} with parameter {param_values}.")
        return transform_dicts

    def get_single_default_transform(self, transform_name, use_minimal=False):
        transform = self.transforms[transform_name]
        params = transform['param_values_minimal'] if use_minimal else transform['param_values']
        return self.get_multiple_transforms(transform_name, transform['param_name'], params)

    def get_all_default_transforms(self, use_minimal=False):
        all_transforms = []
        for transform_name, transform in self.transforms.items():
            params = transform['param_values_minimal'] if use_minimal else transform['param_values']
            transforms = self.get_multiple_transforms(transform_name, transform['param_name'], params)
            all_transforms.extend(transforms)
        return all_transforms

    def _combine_pair_of_transforms(self, transform_i, transform_j, delimiter='~'):
        """ Combine two transforms into a single dictionary. """
        combined_transform = {
            'type': f"{transform_i['type']}{delimiter}{transform_j['type']}",
            'param': f"{transform_i['param']}{delimiter}{transform_j['param']}",
            'param_name': f"{transform_i['param_name']}{delimiter}{transform_j['param_name']}",
            'transform': Compose([transform_i['transform'], transform_j['transform']])
        }
        return combined_transform

    def get_pair_combinations_of_default_transforms(self):
        """ Build similar dictionary as get_all_default_transforms, but with all possible pair combinations of transforms. """
        all_transforms = self.get_all_default_transforms()
        all_transforms_sorted = {key: [] for key in self.transforms.keys()}
        for transform in all_transforms:
            all_transforms_sorted[transform['type']].append(transform)
        transform_pair_combinations = []
        for i, transform_name_i in enumerate(self.transforms.keys()):
            for j, transform_name_j in enumerate(self.transforms.keys()):
                if i >= j:
                    continue
                transforms_list_i = all_transforms_sorted[transform_name_i]
                transforms_list_j = all_transforms_sorted[transform_name_j]
                max_len = max(len(transforms_list_i), len(transforms_list_j))
                for param_idx in range(max_len):
                    param_i = min(param_idx, len(transforms_list_i) - 1)
                    param_j = min(param_idx, len(transforms_list_j) - 1)
                    combined_pair = self._combine_pair_of_transforms(
                        transforms_list_i[param_i],
                        transforms_list_j[param_j]
                    )
                    transform_pair_combinations.append(combined_pair)
        print()
        return transform_pair_combinations



if __name__ == '__main__':
    from sanity_checks.check_transforms import test_all_transforms, test_transform
    pairs = TransformFactory().get_pair_combinations_of_default_transforms()#
    dataset_name = 'imagenet'
    for single_pair in pairs:
        transform_obj = single_pair['transform']
        transform_name = single_pair['type']
        param_name = single_pair['param_name']
        param = single_pair['param']
        path_name = 'pair/' + transform_name
        test_transform(transform_obj, path_name, param, dataset_name)
