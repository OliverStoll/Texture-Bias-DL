import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import tensorboard_trace_handler, schedule

from src.helper.config_loader import config as c
from src.lightning_tutorial.callbacks import MyPrintingCallback
from src.lightning_tutorial.dataset import MNistDataModule
from src.lightning_tutorial.model import NN

LOG_DIR = "../../logs/tb_logs"


class TrainingRunner:
    """Wrapper class to handle data loading, model creation and training."""

    def __init__(self):
        self.logger = TensorBoardLogger(LOG_DIR, name="mnist_model")
        self.profiler = PyTorchProfiler(
            on_trace_ready=tensorboard_trace_handler("tb_logs/profiler"),
            trace_memory=True,
            schedule=schedule(skip_first=2, wait=1, warmup=1, active=20),
        )
        self.model = NN(c["LAYER_SIZES"], c["NUM_CLASSES"], c["LEARNING_RATE"]).to(c["DEVICE"])
        self.data_module = MNistDataModule("dataset/", batch_size=c["BATCH_SIZE"], num_workers=4)
        self.trainer = pl.Trainer(
            logger=self.logger,
            # profiler=self.profiler,
            max_epochs=c["NUM_EPOCHS"],
            precision="bf16-mixed",
            callbacks=[TQDMProgressBar(refresh_rate=30), MyPrintingCallback()],
            fast_dev_run=False,
        )

    def run(self):
        self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.run()
