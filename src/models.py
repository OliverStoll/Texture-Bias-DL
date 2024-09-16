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

    def split_layer_name(self, layer_name):
        """Split a layer name using '.' and '[' as separators. Returns a list of attribute parts."""
        parts = []
        current_part = ''
        for char in layer_name:
            if char == '.' or char == '[' or char == ']':
                if current_part:
                    parts.append(current_part)
                    current_part = ''
            else:
                current_part += char
        if current_part:
            parts.append(current_part)
        return parts

    def get_layer(self, model, layer_name):
        name_parts = self.split_layer_name(layer_name)
        layer = getattr(model, name_parts[0])
        for part in name_parts[1:]:
            if part.isdigit():  # If layer is a list
                layer = layer[int(part)]
            else:               # If layer is an attribute
                layer = getattr(layer, part)
        return layer

    def set_layer(self, model, layer_name, layer):
        """Set nested attributes and indices using a string path."""
        name_parts = self.split_layer_name(layer_name)
        temp_layer = model
        try:
            for part in name_parts[:-1]:
                if part.isdigit():
                    temp_layer = temp_layer[int(part)]
                else:
                    temp_layer = getattr(temp_layer, part)
            last_attribute = name_parts[-1]
            if last_attribute.isdigit():
                temp_layer[int(last_attribute)] = layer
            else:
                setattr(temp_layer, last_attribute, layer)
        except (AttributeError, IndexError, TypeError) as e:
            raise AttributeError(f"Failed to set attribute '{layer_name}': {e}")

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
