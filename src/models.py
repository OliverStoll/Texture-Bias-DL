import timm
from common_utils.logger import create_logger
from common_utils.config import CONFIG


class ModelFactory:
    log = create_logger('ModelFactory')
    default_in_channels = 3
    default_out_features = 1000
    cnn_models = {
        'resnet': 'resnet50',  # 25.6M
        'efficientnet': 'efficientnet_b5',  # 30M
        'convnext': 'convnext_tiny',  # 28M
        'regnet': 'regnetx_016',  # 16M
        'densenet': 'densenet161',  # 28M
        'resnext': 'resnext50_32x4d',  # 25M
        'mobilenet': 'mobilenetv3_large_100',  # 5.5M
        'xception': 'legacy_xception',  # 22.9M
        'inception': 'inception_v3',  # 27M
        'regnety': 'regnety_004',  # 20M
    }
    transformer_models = {
        "vit": "vit_tiny_patch16_224",  # 5.7M
        "deit": "deit_tiny_patch16_224",  # 5.7M
        "swin": "swin_tiny_patch4_window7_224",  # 28M
        "cait": "cait_s24_224",  # 24M
        "pvt": "pvt_v2_b2",  # 25.4M
        "pit": "pit_ti_224",  # 4.9M
        'beit': 'beit_base_patch16_224',  # 86M TODO: TOO BIG
        'convmixer': 'convmixer_768_32',  # 21M
        'mvit': 'mvitv2_tiny',  # 25M
    }
    all_models = {**cnn_models, **transformer_models}
    # model names for external use
    cnn_model_names = list(cnn_models.keys())
    transformer_model_names = list(transformer_models.keys())
    all_model_names = list(all_models.keys())
    example_cnn_model_names = ['resnet', 'efficientnet', 'densenet', 'mobilenet']
    example_transformer_model_names = ['vit', 'deit', 'swin', 'pvt']
    example_model_names = example_cnn_model_names + example_transformer_model_names
    # models that need image size as input
    img_size_models = list(transformer_models.keys())
    for no_img_size_transformer in ['pvt']:
        img_size_models.remove(no_img_size_transformer)


    def get_in_out_channels(self, data_conf):
        in_channels = data_conf['input_channels']
        in_channels = in_channels if in_channels != self.default_in_channels else None
        if data_conf['task'] == 'multiclass':
            out_features = data_conf['num_classes']
        elif data_conf['task'] == 'multilabel':
            out_features = data_conf['num_labels']
        else:
            raise ValueError(f"Invalid task type: {data_conf['task']}")
        out_features = out_features if out_features != self.default_out_features else None
        return in_channels, out_features

    def get_model(self, model_name: str, dataset_config: dict | str, pretrained: bool):
        if isinstance(dataset_config, str):
            dataset_config = CONFIG['datasets'][dataset_config]
        full_model_name = self.all_models[model_name]
        in_channels, out_features = self.get_in_out_channels(dataset_config)
        img_size = dataset_config['image_size'] if model_name in self.img_size_models else None
        self.log.debug(f"Initializing model: [{model_name}{' | Pretrained' if pretrained else ''}] ")
        model = timm.create_model(
            model_name=full_model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=out_features,
            img_size=img_size
        )
        return model



if __name__ == "__main__":
    model_initializer = ModelFactory()
    dataset_config = {'num_classes': 10, 'task': 'multiclass', 'input_channels': 3,
                      'image_size': 224}
    ben_data_conf = {'num_labels': 12, 'task': 'multilabel', 'input_channels': 14,
                     'image_size': 224}
    for no_img_size_transformer in model_initializer.all_models.keys():
        print(no_img_size_transformer)
        model_initializer.get_model(no_img_size_transformer, ben_data_conf, pretrained=False)
