import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0, vit_b_16, swin_t, convnext_tiny
from torchvision.models import (ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
                                Swin_T_Weights, ConvNeXt_Tiny_Weights)


class ModelCollection:
    def __init__(self):
        """ Changes to the model collection also need to be applied in data_init.py ! """
        self.models_dict = {
            'resnet': {'init_fn': self.init_resnet50, 'weights': ResNet50_Weights},
            'efficientnet': {'init_fn': self.init_efficientnet_b0,
                             'weights': EfficientNet_B0_Weights},
            'convnext': {'init_fn': self.init_convnext_tiny, 'weights': ConvNeXt_Tiny_Weights},
            'vit': {'init_fn': self.init_vit_b_16, 'weights': ViT_B_16_Weights},
            'swin': {'init_fn': self.init_swin_transformer_t, 'weights': Swin_T_Weights},
        }

    def get_model(self, model_name: str, dataset_config: dict, pretrained: bool):
        model_dict = ModelCollection().models_dict
        weights = model_dict[model_name]['weights'].DEFAULT if pretrained else None
        model_init_function = model_dict[model_name]['init_fn']
        model = model_init_function(dataset_config, weights)
        return model

    def _get_num_classes(self, data_conf):
        return data_conf['num_classes'] if data_conf['task'] == 'multiclass' else data_conf[
            'num_labels']

    def init_resnet50(self, data_conf, weights):
        resnet_model = resnet50(weights=weights)
        if data_conf['input_channels'] != 3:
            resnet_model.conv1 = nn.Conv2d(in_channels=data_conf['input_channels'],
                                           out_channels=64,
                                           kernel_size=7,
                                           stride=2,
                                           padding=3,
                                           bias=False)
        if self._get_num_classes(data_conf) != 1000:
            resnet_model.fc = nn.Linear(in_features=resnet_model.fc.in_features,
                                        out_features=self._get_num_classes(data_conf))
        return resnet_model

    def init_efficientnet_b0(self, data_conf, weights):
        efnet_model = efficientnet_b0(weights=weights)
        if data_conf['input_channels'] != 3:
            efnet_model.features[0][0] = nn.Conv2d(in_channels=data_conf['input_channels'],
                                                   out_channels=32,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   bias=False)
        if self._get_num_classes(data_conf) != 1000:
            efnet_model.classifier[1] = nn.Linear(in_features=efnet_model.classifier[1].in_features,
                                                  out_features=self._get_num_classes(data_conf))
        return efnet_model

    def init_convnext_tiny(self, data_conf, weights):
        conv_model = convnext_tiny(weights=weights)
        if data_conf['input_channels'] != 3:
            conv_model.features[0][0] = nn.Conv2d(in_channels=data_conf['input_channels'],
                                                  out_channels=96,
                                                  kernel_size=4,
                                                  stride=4)
        if self._get_num_classes(data_conf) != 1000:
            conv_model.classifier[2] = nn.Linear(in_features=conv_model.classifier[2].in_features,
                                                 out_features=self._get_num_classes(data_conf))
        return conv_model

    def init_vit_b_16(self, data_conf, weights):
        vit_model = vit_b_16(weights=weights, image_size=data_conf['image_size'])
        if data_conf['input_channels'] != 3:
            vit_model.conv_proj = nn.Conv2d(in_channels=data_conf['input_channels'],
                                            out_channels=vit_model.conv_proj.out_channels,
                                            kernel_size=16,
                                            stride=16)
        if self._get_num_classes(data_conf) != 1000:
            vit_model.heads.head = nn.Linear(in_features=vit_model.heads.head.in_features,
                                             out_features=self._get_num_classes(data_conf))
        return vit_model

    def init_swin_transformer_t(self, data_conf, weights):
        swin_model = swin_t(weights=weights)
        if data_conf['input_channels'] != 3:
            swin_model.features[0][0] = nn.Conv2d(in_channels=data_conf['input_channels'],
                                                  out_channels=96,
                                                  kernel_size=4,
                                                  stride=4)
        if self._get_num_classes(data_conf) != 1000:
            swin_model.head = nn.Linear(in_features=swin_model.head.in_features,
                                        out_features=self._get_num_classes(data_conf))
        return swin_model


if __name__ == "__main__":
    model_initializer = ModelCollection()
    dataset_config = {'num_classes': 10, 'task': 'multiclass', 'input_channels': 3, 'image_size': 224}
    for model_name in ['resnet', 'efficientnet', 'convnext', 'vit', 'swin']:
        model, weights_transform_fn = model_initializer.get_model(model_name, dataset_config, pretrained=True)
        print(model)
        print(weights_transform_fn)
        print()
