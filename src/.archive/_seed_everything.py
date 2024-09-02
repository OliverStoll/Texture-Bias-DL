from pytorch_lightning import seed_everything
import logging

# hide the seed logger
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)
seed_everything(seed=42)
