import timm
from common_utils.logger import create_logger
from common_utils.config import CONFIG



class ModelFactory:
    """
    Factory for instantiating image classification models using `timm`.

    Supports CNN and transformer-based backbones with automatic handling of
    input/output dimensions and image size requirements.
    """

    log = create_logger('ModelFactory')

    default_in_channels: int = 3
    default_out_features: int = 1000

    cnn_models: dict[str, str] = {
        'resnet': 'resnet50',
        'efficientnet': 'efficientnet_b5',
        'convnext': 'convnext_tiny',
        'regnet': 'regnetx_016',
        'densenet': 'densenet161',
        'resnext': 'resnext50_32x4d',
        'mobilenet': 'mobilenetv3_large_100',
        'xception': 'legacy_xception',
        'inception': 'inception_v3',
        'regnety': 'regnety_004',
    }

    transformer_models: dict[str, str] = {
        "vit": "vit_tiny_patch16_224",
        "deit": "deit_tiny_patch16_224",
        "swin": "swin_tiny_patch4_window7_224",
        "cait": "cait_s24_224",
        "pvt": "pvt_v2_b2",
        "pit": "pit_ti_224",
        'convmixer': 'convmixer_768_32',
        'mvit': 'mvitv2_tiny',
    }


    all_models: dict[str, str] = {**cnn_models, **transformer_models}
    cnn_model_names: list[str] = list(cnn_models.keys())
    transformer_model_names: list[str] = list(transformer_models.keys())
    all_model_names: list[str] = list(all_models.keys())
    implicit_img_size_models: list[str] = ['pvt']
    explicit_img_size_models: list[str] = [k for k in transformer_models if k not in implicit_img_size_models]

    def _get_in_out_channels(self, data_conf: dict) -> tuple[int | None, int | None]:
        """
        Determine the input and output channels based on dataset configuration.

        Args:
            data_conf: Dictionary containing dataset properties like task type and channel counts.

        Returns:
            Tuple of (in_channels, out_features), where values are None if defaults are used.
        """
        input_channels = data_conf['input_channels']
        input_channels = None if input_channels == self.default_in_channels else input_channels

        task_type = data_conf['task']
        if task_type == 'multiclass':
            out_features = data_conf['num_classes']
        elif task_type == 'multilabel':
            out_features = data_conf['num_labels']
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        out_features = None if out_features == self.default_out_features else out_features

        return input_channels, out_features

    def get_model(self, model_name: str, dataset_config: dict | str, pretrained: bool):
        """
        Instantiate a timm model based on model name and dataset configuration.

        Args:
            model_name: Short key for selecting the model architecture.
            dataset_config: Dataset name or config dictionary containing model input/output specs.
            pretrained: Whether to load pretrained weights.

        Returns:
            An instantiated timm model.
        """
        if isinstance(dataset_config, str):
            dataset_config = CONFIG['datasets'][dataset_config]

        full_model_name = self.all_models[model_name]
        in_channels, out_features = self._get_in_out_channels(dataset_config)
        img_size = dataset_config['image_size'] if model_name in self.explicit_img_size_models else None

        self.log.debug(f"Initializing model: [{model_name}] {'| Pretrained' if pretrained else '| Scratch'}")

        model_kwargs = {
            'model_name': full_model_name,
            'pretrained': pretrained,
            'in_chans': in_channels,
            'num_classes': out_features,
            'img_size': img_size,
        }

        if model_name in {'vit', 'swin', 'mvit'}:
            model_kwargs['patch_size'] = 12

        return timm.create_model(**model_kwargs)


if __name__ == "__main__":
    """ Test script for verifying model instantiation via ModelFactory. """
    model_initializer = ModelFactory()
    dataset_config_ = {'num_classes': 10, 'task': 'multiclass', 'input_channels': 3,
                      'image_size': 224}
    ben_data_conf = {'num_labels': 12, 'task': 'multilabel', 'input_channels': 14,
                     'image_size': 224}
    for no_img_size_transformer in model_initializer.all_models.keys():
        print(no_img_size_transformer)
        model_initializer.get_model(no_img_size_transformer, ben_data_conf, pretrained=False)
