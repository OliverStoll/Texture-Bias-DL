from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    """Simple callback to notify when training starts and ends."""

    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done!")
