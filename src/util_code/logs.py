import os
import logging
import warnings


def mute_logs():
    os.environ['WANDB_SILENT'] = 'true'
    logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning.accelerators.cuda').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="lightning_fabric.plugins.environments.slurm")
    warnings.filterwarnings("ignore",
                            message="Average precision score for one or more classes was `nan`. "
                                    "Ignoring these classes in weighted-average")