from val_transforms.grid_shuffle import GridShuffleTransform
from val_transforms.low_pass import LowPassFilterTransform
from val_transforms.edge_detection import EdgeDetectionTransform
from val_transforms.bilateral import BilateralFilterTransform
from utils.logger import create_logger


class TransformFactory:
    log = create_logger("Transforms")

    def __init__(self):
        self.transforms_classes = {
            'grid_shuffle': GridShuffleTransform,
            'low_pass': LowPassFilterTransform,
            'edges': EdgeDetectionTransform,
            'bilateral': BilateralFilterTransform,
        }

    def get_transforms(self, transform_type, param_name, param_values):
        # todo: add multi-parameter support
        transform_dicts = [
            {
                'type': transform_type,
                'param': param,
                'param_name': param_name,
                param_name: param,  # redundant
                'transform': self.transforms_classes[transform_type](**{param_name: param})
            } for param in param_values
        ]
        self.log.info(f"Created {len(transform_dicts)} transforms of type {transform_type} with parameter {param_values}.")
        return transform_dicts


empty_transforms = [{
    'type': None,
    'param': None,
    'param_name': None,
    'transform': None,
}]