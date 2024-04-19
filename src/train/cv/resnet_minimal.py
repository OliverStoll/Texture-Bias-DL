import torchmetrics
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import tensorboard_trace_handler, schedule

# local imports
from utils.config import ROOT_DIR, CONFIG as config
from data_loading.imagenet.ImageNetDataModule import ImageNetDataModule


class PretrainedResnet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, num_classes)
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        return self.resnet_model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.f1_score(scores, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            {"test_f1": self.f1_score.compute()}, on_epoch=True, prog_bar=True, logger=True
        )



LOG_DIR = "../../logs/tb_logs"


class TrainingRunner:
    """Wrapper class to handle data loading, model creation and training."""

    def __init__(self):
        print(f'{ROOT_DIR}logs/wandb')
        self.logger = WandbLogger(project="Resnet-?", save_dir=f'{ROOT_DIR}logs')
        self.profiler = PyTorchProfiler(
            on_trace_ready=tensorboard_trace_handler("tb_logs/profiler"),
            trace_memory=True,
            schedule=schedule(skip_first=2, wait=1, warmup=1, active=20),
        )
        self.model = PretrainedResnet(config["NUM_CLASSES"]).to(config["DEVICE"])
        self.data_module = ImageNetDataModule(
            data_dir=config["DATASET_PATH"], batch_size=config["BATCH_SIZE"], num_workers=4
        )
        self.trainer = pl.Trainer(
            logger=self.logger,
            # profiler=self.profiler,
            max_epochs=config["NUM_EPOCHS"],
            precision="bf16-mixed",
            callbacks=[TQDMProgressBar(refresh_rate=30)],
            fast_dev_run=False,
        )

    def run(self):
        # self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.run()