from torchvision.transforms import Compose
from common_utils.logger import create_logger

from transforms.transform_functions.grid_shuffle import PatchShuffleTransform
from transforms.transform_functions.sobel import SobelFilterTransform
from transforms.transform_functions.bilateral import BilateralFilterTransform
from transforms.transform_functions.median import MedianFilterTransform
from transforms.transform_functions.gaussian_blur import GaussianBlurTransform
from transforms.transform_functions.channel_shuffle import ChannelShuffleTransform
from transforms.transform_functions.channel_inversion import ChannelInversionTransform
from transforms.transform_functions.grayscale import GrayScaleTransform
from transforms.transform_functions.patch_rotation import PatchRotationTransform
from transforms.transform_functions.noise import NoiseFilterTransform

empty_transforms = [{
    'type': None,
    'param': None,
    'param_name': None,
    'transform': None,
}]


class TransformFactory:
    log = create_logger("TransformFactory")
    transform_combinations = [
        ('bilateral', 'patch_shuffle'),
    ]

    def __init__(self):
        self.transforms = {
            'Bilateral': {
                'transform_type': 'texture',
                'class': BilateralFilterTransform,
                'param_name': 'd',
                'param_values': [0, 3, 5, 7, 9, 11, 15],
            },
            'Median': {
                'transform_type': 'texture',
                'class': MedianFilterTransform,
                'param_name': 'kernel_size',
                'param_values': [0, 3, 5, 7, 9, 11, 15],
            },
            'Gaussian': {
                'transform_type': 'texture',
                'class': GaussianBlurTransform,
                'param_name': 'sigma',
                'param_values': [0., 1., 2., 3., 4., 5., 7.],
            },
            'Patch Shuffle': {
                'transform_type': 'shape',
                'class': PatchShuffleTransform,
                'param_name': 'grid_size',
                'param_values': [0, 2, 4, 6, 8, 11, 15],
            },
            'Patch Rotation': {
                'transform_type': 'shape',
                'class': PatchRotationTransform,
                'param_name': 'grid_size',
                'param_values': [0, 2, 4, 6, 8, 11, 15],
            },
            'Channel Shuffle': {
                'transform_type': 'color',
                'class': ChannelShuffleTransform,
                'param_name': 'n',
                'param_values': [0, 2, 3, 5, 7, 9, 12],
            },
            'Channel Inversion': {
                'transform_type': 'color',
                'class': ChannelInversionTransform,
                'param_name': 'n',
                'param_values': [0, 1, 2, 3, 6, 9, 12],
            },
            'Channel Mean': {
                'transform_type': 'color',
                'class': GrayScaleTransform,
                'param_name': 'percentage',
                'param_values': [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1],
            },
            'Noise': {
                'transform_type': 'shape',
                'class': NoiseFilterTransform,
                'param_name': 'intensity',
                'param_values': [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1],
            },
        }
        self.paired_keys = ['Bilateral', 'Patch Shuffle', 'Channel Shuffle']


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

    def get_single_default_transform(self, transform_name):
        transform = self.transforms[transform_name]
        return self.get_multiple_transforms(transform_name, transform['param_name'], transform['param_values'])

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
        for i, transform_name_i in enumerate(self.paired_keys):
            for j, transform_name_j in enumerate(self.paired_keys):
                if i >= j:
                    continue
                transform_type_i = self.transforms[transform_name_i]['transform_type']
                transform_type_j = self.transforms[transform_name_j]['transform_type']
                if transform_type_i == transform_type_j:
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
        self.log.warning(f"TESTING | CREATED PAIRS: {transform_pair_combinations}")
        return transform_pair_combinations




if __name__ == '__main__':
    from tests.transforms import test_transform
    img_idx = {
        'imagenet': 1,
        'deepglobe': 1,
    }
    neutral_transform = TransformFactory().get_single_default_transform('Bilateral')[0]
    dataset_names = ['imagenet', 'bigearthnet', 'caltech', 'deepglobe']
    dataset_names = ['imagenet', 'caltech', 'deepglobe']
    dataset_names = ['deepglobe']
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


        # plot 10 examples:
        try:
            for i in range(0, 10000, 100):
                test_transform(
                    transform=neutral_transform['transform'],
                    transform_name=f"ORIGINAL",
                    param=i,
                    dataset=dataset_name,
                    example_idx=i
                )
        except Exception as e:
            for i in range(0, 4):
                test_transform(
                    transform=neutral_transform['transform'],
                    transform_name=f"ORIGINAL",
                    param=i,
                    dataset=dataset_name,
                    example_idx=i
                )


