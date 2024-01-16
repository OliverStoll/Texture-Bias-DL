import pytorch_lightning as pl

from src.helper.config_loader import config
from src.lightning_tutorial.callbacks import MyPrintingCallback
from src.lightning_tutorial.dataset import MNistDataModule
from src.lightning_tutorial.model import NN


class TrainingRunner:
    """Wrapper class to handle data loading, model creation and training."""

    def __init__(self):
        self.model = NN(config["LAYER_SIZES"], config["NUM_CLASSES"], config["LEARNING_RATE"]).to(
            config["DEVICE"]
        )
        self.dm = MNistDataModule(
            data_dir="dataset/", batch_size=config["BATCH_SIZE"], num_workers=4
        )
        self.trainer = pl.Trainer(
            max_epochs=config["NUM_EPOCHS"],
            precision="bf16-mixed",
            callbacks=[
                pl.callbacks.progress.TQDMProgressBar(refresh_rate=30),
                MyPrintingCallback(),
            ],
            fast_dev_run=False,
        )

    def train(self):
        self.trainer.fit(self.model, self.dm)


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.train()
