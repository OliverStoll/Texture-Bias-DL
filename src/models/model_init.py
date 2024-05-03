import torch.nn as nn
from torchvision.models import resnet18, resnet50
# import resnet weights
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights


def init_resnet18(dataset_config, pretrained):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    resnet_model = resnet18(weights=weights)
    resnet_model.conv1 = nn.Conv2d(dataset_config['input_channels'], out_channels=64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
    num_classes = dataset_config['num_classes'] if 'num_classes' in dataset_config else dataset_config['num_labels']
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model


def init_resnet50(dataset_config, pretrained):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    resnet_model = resnet50(weights=weights)
    resnet_model.conv1 = nn.Conv2d(dataset_config['input_channels'], out_channels=64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
    num_classes = dataset_config['num_classes'] if dataset_config['task'] == 'multiclass' else dataset_config['num_labels']
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model


# TODO: Add inits for other models

def get_model(model_name: str, dataset_config: dict, pretrained: bool):
    match model_name:
        case 'resnet18':
            return init_resnet18(dataset_config, pretrained)
        case 'resnet50':
            return init_resnet50(dataset_config, pretrained)
        case 'resnet':
            return init_resnet50(dataset_config, pretrained)
        case _:
            raise ValueError(f"Model {model_name} not supported")
