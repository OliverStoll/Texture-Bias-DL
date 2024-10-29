from runs import RunManager
from transforms import TransformFactory
from models import ModelFactory
from datasets import DataLoaderFactory
import torch
import json
import numpy as np
# import transforms
from torchvision import transforms

from common_utils.config import CONFIG


if __name__ == '__main__':
    train_dl, _, _ = DataLoaderFactory().get_dataloader('bigearthnet', train_transform=transforms.Compose([]))
    first_batch = next(iter(train_dl))
    # save first image to disk
    first_img_tensor = first_batch[0][0]
    # save tensor to file
    torch.save(first_img_tensor, CONFIG['example_BEN_tensor_unnormalized'])

