import os

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
from pathlib import Path
import torch


data_dirs = {
            "images_lmdb": Path("/faststorage") / "BigEarthNet-V2" / "BigEarthNet-V2-LMDB",
            "metadata_parquet": Path("/faststorage") / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/faststorage")
            / "BigEarthNet-V2"
            / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        }

ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=data_dirs
)

output_path = "output"
os.makedirs(output_path, exist_ok=True)
for i in range(0, 10000, 100):
    img, lbl = ds[i]
    print(img)
    # make label human readable
    lbl = torch.where(lbl == 1)[0]
    # save tensor
    lbl = ",".join([str(i.item()) for i in lbl])
    print(i, lbl)
    torch.save(img, f"{output_path}/bigearthnet_{i}.pt")