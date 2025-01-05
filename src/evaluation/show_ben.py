from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
from pathlib import Path

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

img, lbl = ds[0]
print(img.shape, lbl.shape)

# save image
import PIL
from PIL import Image
img = Image.fromarray(img)
img.save("test.png")