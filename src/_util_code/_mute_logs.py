import os
import logging


os.environ['WANDB_SILENT'] = 'true'
logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)
logging.getLogger('pytorch_lightning.accelerators.cuda').setLevel(logging.WARNING)