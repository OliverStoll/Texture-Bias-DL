import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from common_utils.config import CONFIG


to_tensor = ToTensor()
to_image = ToPILImage()

ben_mean = {
    "B01": 361.0660705566406,
    "B02": 438.385009765625,
    "B03": 614.0690307617188,
    "B04": 588.4198608398438,
    "B05": 942.854248046875,
    "B06": 1769.951904296875,
    "B07": 2049.574951171875,
    "B08": 2193.320556640625,
    "B09": 2241.467529296875,
    "B11": 1568.229736328125,
    "B12": 997.7351684570312,
    "B8A": 2235.580810546875,
}
ben_std = {
    "B01": 575.0419311523438,
    "B02": 607.0984497070312,
    "B03": 603.3590087890625,
    "B04": 684.6243896484375,
    "B05": 738.4649047851562,
    "B06": 1100.47900390625,
    "B07": 1275.8310546875,
    "B08": 1369.4029541015625,
    "B09": 1316.40380859375,
    "B11": 1070.170166015625,
    "B12": 813.5411376953125,
    "B8A": 1356.568359375,
}
ben_mean_list = [ben_mean[key] for key in sorted(ben_mean.keys())]
ben_std_list = [ben_std[key] for key in sorted(ben_std.keys())]
BEN_CHANNELS = (3, 2, 1)

# TODO: add mean & std for caltech & deepglobe
MEANS = {
    'bigearthnet': ben_mean_list,
    'imagenet': [0.485, 0.456, 0.406],
    'caltech': [],
    'deepglobe': []
}
STDS = {
    'bigearthnet': ben_std_list,
    'imagenet': [0.229, 0.224, 0.225],
    'caltech': [],
    'deepglobe': []
}


def get_example_image():
    img = Image.open(CONFIG['example_image'])
    return img


def get_example_tensor(dataset):
    tensor_path = f"{CONFIG['example_tensors_path']}/{dataset}.pt"
    tensor = torch.load(tensor_path)
    return tensor


def denormalize_tensor(tensor, dataset):
    mean = MEANS[dataset]
    std = STDS[dataset]
    for tensor_channel, mean, std in zip(tensor, mean, std):
        tensor_channel.mul_(std).add_(mean)
    return tensor


def show_original_example_image(datasets=('bigearthnet', 'imagenet')):
    for dataset in datasets:
        test_tensor = get_example_tensor(dataset=dataset)
        test_tensor_un = denormalize_tensor(test_tensor, dataset)
        if dataset == 'bigearthnet':
            test_tensor_un = test_tensor_un[BEN_CHANNELS, :, :]
        test_image = to_image(test_tensor_un)
        test_image.save(f"{CONFIG['example_image_output']}/{dataset}_original.png")


def test_transform(transform, transform_name, param, dataset):
    save_path = f"{CONFIG['example_image_output']}/{transform_name}"
    os.makedirs(save_path, exist_ok=True)
    image_tensor = get_example_tensor(dataset)
    transformed_tensor = transform(image_tensor)
    transformed_tensor = denormalize_tensor(transformed_tensor, dataset)
    if dataset == 'bigearthnet':
        transformed_tensor = transformed_tensor[BEN_CHANNELS, :, :]
    # switch channels inside first dimension from bgr to rgb
    correct_tuple = (0, 1, 2) # gelb-grün
    correct_tuple = (2, 1, 0) # türkis
    correct_tuple = (2, 0, 1) # blau
    correct_tuple = (1, 2, 0) # lila
    correct_tuple = (1, 0, 2) # gelb-braun
    correct_tuple = (0, 2, 1) # lila-weiß?
    correct_tuple = (1, 2, 0)  # lila
    transformed_tensor = transformed_tensor[correct_tuple, :, :]
    transformed_image = to_image(transformed_tensor)
    example_img_path = f"{save_path}/{dataset}_{param}.png"
    print(f"saving image to {example_img_path}")
    transformed_image.save(example_img_path)



if __name__ == '__main__':
    show_original_example_image()


