from configilm.extra.DataModules import BENv2DataModule
from configilm.extra.data_dir import dataset_paths

from utils.config import CONFIG


benv2_mapping = dataset_paths['benv2'][1]  # 1 for erde server
datamodule = BENv2DataModule(
    data_dirs=benv2_mapping,
    batch_size=CONFIG['batch_size'],
    num_workers_dataloader=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory'],
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
item = iter(train_loader).next()
print()


