import pytorch_lightning as pl

# local
from src.lightning_tutorial.callbacks import MyPrintingCallback
from src.lightning_tutorial.config import (
    num_classes,
    learning_rate,
    batch_size,
    num_epochs,
    layer_sizes,
    device,
)
from src.lightning_tutorial.dataset import MNistDataModule
from src.lightning_tutorial.model import NN

if __name__ == "__main__":
    model = NN(layer_sizes, num_classes, learning_rate).to(device)
    dm = MNistDataModule(data_dir="dataset/", batch_size=batch_size, num_workers=4)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        precision="bf16-mixed",
        callbacks=[
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=30),
            MyPrintingCallback(),
        ],
        fast_dev_run=False,
    )
    trainer.fit(model, dm)
