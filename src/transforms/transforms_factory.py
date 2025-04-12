from torchvision.transforms import Compose
from common_utils.logger import create_logger

from transforms.transform_functions.grid_shuffle import PatchShuffleTransform
from transforms.transform_functions.bilateral import BilateralFilterTransform
from transforms.transform_functions.median import MedianFilterTransform
from transforms.transform_functions.gaussian_blur import GaussianBlurTransform
from transforms.transform_functions.channel_shuffle import ChannelShuffleTransform
from transforms.transform_functions.channel_inversion import ChannelInversionTransform
from transforms.transform_functions.grayscale import GrayScaleTransform
from transforms.transform_functions.patch_rotation import PatchRotationTransform
from transforms.transform_functions.noise import NoiseFilterTransform


empty_transform = {
    'type': None,
    'param': None,
    'param_name': None,
    'transform': None
}


class TransformFactory:
    """
    Factory class to manage feature suppression transforms for shape, texture, and color.
    Allows customization of the transform set and pairing behavior for evaluation of two suppressed image features.
    """

    log = create_logger("TransformFactory")
    transforms = {
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
    paired_keys = ['Bilateral', 'Patch Shuffle', 'Channel Shuffle']

    def __init__(
            self,
            paired_keys: list[str] | None = None,
            transforms: dict[str, dict] | None = None
    ) -> None:
        """
        Initialize the TransformFactory with optional custom transform configurations.

        Args:
            paired_keys: List of transform names for two feature suppression evaluation.
            transforms: Dictionary specifying available transforms for custom intensity application.
        """
        if paired_keys is not None:
            self.paired_keys = paired_keys
        if transforms is not None:
            self.transforms = transforms


    def get_multiple_transforms(
            self,
            transform_name: str,
            param_name: str | None = None,
            param_values: list | float | int | None = None
    ) -> list[dict]:
        """
        Create a list of transform configurations for a given transform type and parameter values.

        Args:
            transform_name: Name of the transform to instantiate (must be defined in self.transforms).
            param_name: The parameter name to vary. If None, uses the default from the transform config.
            param_values: List of parameter values or a single value. If None, uses default values.

        Returns:
            A list of dictionaries, each representing a transform instance with metadata.
        """
        if param_name is not None and param_values is None:
            raise ValueError("If param_name is provided, param_values must be provided as well.")

        param_name = param_name or self.transforms[transform_name]['param_name']
        param_values = param_values or self.transforms[transform_name]['param_values']

        if not isinstance(param_values, list):
            param_values = [param_values]

        transform_class = self.transforms[transform_name]['class']
        transform_dicts = [
            {
                'type': transform_name,
                'param': param,
                'param_name': param_name,
                'transform': transform_class(**{param_name: param})
            }
            for param in param_values
        ]
        self.log.debug(
            f"Created {len(transform_dicts)} transforms of type {transform_name} "
            f"with {param_name} parameter {param_values}."
        )

        return transform_dicts

    def get_single_default_transform(self, transform_name: str) -> list[dict]:
        """
        Return the default set of transforms for a given transform name
        using its configured parameter name and default parameter values.

        Args:
            transform_name: Name of the transform to instantiate.

        Returns:
            A list of dictionaries representing instantiated transforms.
        """
        transform_config = self.transforms[transform_name]
        return self.get_multiple_transforms(
            transform_name=transform_name,
            param_name=transform_config['param_name'],
            param_values=transform_config['param_values']
        )


    def get_all_default_transforms(self) -> list[dict]:
        """
        Retrieve default transform instances for all defined transform types.

        Returns:
            A list of dictionaries, each representing an instantiated transform with metadata.
        """
        all_transforms = []
        for transform_name, transform in self.transforms.items():
            transforms = self.get_multiple_transforms(
                transform_name=transform_name,
                param_name=transform['param_name'],
                param_values=transform['param_values']
            )
            all_transforms.extend(transforms)

        return all_transforms

    @staticmethod
    def _combine_pair_of_transforms(
        transform_i: dict,
        transform_j: dict,
        delimiter: str = '~'
    ) -> dict:
        """
        Combine two transform specifications into a single composite transform.

        Args:
            transform_i: First transform dictionary.
            transform_j: Second transform dictionary.
            delimiter: Delimiter to join type, param, and param_name fields.

        Returns:
            A new transform dictionary with composed transform and merged metadata.
        """
        combined_transform = {
            'type': f"{transform_i['type']}{delimiter}{transform_j['type']}",
            'param': f"{transform_i['param']}{delimiter}{transform_j['param']}",
            'param_name': f"{transform_i['param_name']}{delimiter}{transform_j['param_name']}",
            'transform': Compose([
                transform_i['transform'],
                transform_j['transform']
            ])
        }

        return combined_transform

    @staticmethod
    def _combine_two_transforms(
            self,
            all_transforms_sorted: dict[str, list[dict]],
            name_i: str,
            name_j: str
    ) -> list[dict]:
        """
        Combine two sets of transforms by matching parameter values index-wise (with clipping),
        producing composed transform pairs.

        Args:
            all_transforms_sorted: A dictionary mapping transform names to their instantiated variants.
            name_i: Name of the first transform to combine.
            name_j: Name of the second transform to combine.

        Returns:
            A list of combined transform dictionaries.
        """
        transform_pair_combinations = []
        transform_params_i = all_transforms_sorted[name_i]
        transform_params_j = all_transforms_sorted[name_j]
        max_len = max(len(transform_params_i), len(transform_params_j))
        for param_idx in range(max_len):
            param_i = min(param_idx, len(transform_params_i) - 1)
            param_j = min(param_idx, len(transform_params_j) - 1)
            combined_pair = self._combine_pair_of_transforms(
                transform_params_i[param_i],
                transform_params_j[param_j]
            )
            transform_pair_combinations.append(combined_pair)
        return transform_pair_combinations

    def get_pair_combinations_of_default_transforms(self) -> list[dict]:
        """
        Build a list of transform dictionaries, each representing a pairwise composition of
        transforms from `self.paired_keys`, where each pair combines transforms of different types.

        Returns:
            A list of composed transform dictionaries (with metadata).
        """
        all_transforms = self.get_all_default_transforms()
        all_transforms_sorted = {key: [] for key in self.transforms.keys()}
        for transform in all_transforms:
            all_transforms_sorted[transform['type']].append(transform)
        all_transform_pair_combinations = []
        for i, name_i in enumerate(self.paired_keys):
            for j, name_j in enumerate(self.paired_keys):
                type_i = self.transforms[name_i]['transform_type']
                type_j = self.transforms[name_j]['transform_type']
                if type_i == type_j or i >= j:
                    continue
                all_transform_pair_combinations += self._combine_two_transforms(all_transforms_sorted, name_i, name_j)

        self.log.warning(f"TESTING | CREATED PAIRS: {all_transform_pair_combinations}")
        return all_transform_pair_combinations

