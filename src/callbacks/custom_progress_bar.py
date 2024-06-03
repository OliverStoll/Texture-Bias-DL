from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm
import sys


class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()  # Initialize the base ProgressBar class

    def init_validation_tqdm(self):
        # Always return a disabled tqdm instance for validation
        return tqdm(disable=True)

    def init_train_tqdm(self):
        # Enable the tqdm instance for training
        return tqdm(total=self.total_train_batches)
