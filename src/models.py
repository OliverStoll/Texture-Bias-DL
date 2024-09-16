import timm
import torch.nn as nn


class ModelCollection:
    default_in_channels = 3
    default_out_features = 1000
    model_dict = {
        'resnet': {'name': 'resnet50', 'first_conv': 'conv1', 'out_layer': 'fc'},
        'efficientnet': {'name': 'efficientnet_b0', 'first_conv': 'conv_stem', 'out_layer': 'classifier'},
        'convnext': {'name': 'convnext_tiny', 'first_conv': 'stem[0]', 'out_layer': 'head.fc'},
        'vit': {'name': 'vit_base_patch16_224', 'first_conv': 'patch_embed.proj', 'out_layer': 'head'},
        'swin': {'name': 'swin_tiny_patch4_window7_224', 'first_conv': 'patch_embed.proj', 'out_layer': 'head.fc'},
    }

    def get_in_out_channels(self, model, data_conf):
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

    def get_model(self, model_name: str, data_conf: dict, pretrained: bool):
        architecture = self.model_dict[model_name]['name']
        in_channels, out_features = self.get_in_out_channels(model_name, data_conf)
        model = timm.create_model(architecture, pretrained=pretrained, num_classes=out_features, in_chans=in_channels)

        return model



if __name__ == "__main__":
    model_initializer = ModelCollection()
    dataset_config = {'num_classes': 10, 'task': 'multiclass', 'input_channels': 3,
                      'image_size': 224}
    ben_data_conf = {'num_labels': 12, 'task': 'multilabel', 'input_channels': 14,
                     'image_size': 224}
    for model in model_initializer.model_dict.keys():
        print(model_initializer.get_model(model, ben_data_conf, pretrained=False))
