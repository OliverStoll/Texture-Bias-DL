import torch
import os
import requests
import tarfile
from tqdm import tqdm
from torchvision import models, datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


class ResNetTester(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = torch.sum(torch.argmax(logits, dim=1) == y).float() / len(y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_dataloader(self):
        # Define the data transformations
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Load the ImageNet dataset
        imagenet_test = datasets.ImageFolder("C:/CODE/imagenet/1k-valid", transform=transform)
        return DataLoader(imagenet_test, batch_size=128, shuffle=False)


def download_and_extract_dataset(download_url, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Download the dataset
    filename = os.path.join(destination_path, "imagenet_1k.tar.gz")
    response = requests.get(download_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    # Extract the dataset
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=destination_path)


DATASET_URL = "https://huggingface.co/datasets/imagenet-1k/resolve/main/data/test_images.tar.gz"
DATASET_PATH = "C:/CODE/imagenet/1k-test"

# Initialize the tester
trainer = Trainer()
trainer.test(ResNetTester())
