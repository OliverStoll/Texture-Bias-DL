# BENv2 DataLoading

This repository contains the code for loading the BENv2 dataset in it's optimized form. 
The threadsafe `BENv2LMDBReader` is in the `BENv2Utils.py` file and not dependent on any specific deep learning library. 
It is also not dependent on any other code in this repository.
It is however dependent on the `python-lmdb` package, `pandas` (with `fastparquet` or `pyarrow`), `numpy` and `safetensors`.

The `BENv2DataSet` class in the `BENv2DataSet.py` file is a pure pytorch Dataset that uses the LMDB Reader to load the data. 
It is only dependent on `BENv2Utils.py` (and it's dependencies) and `torch`.

The `BENv2DataModule` class in the `BENv2DataModule.py` file is a pytorch lightning DataModule that uses the `BENv2DataSet` to load the data.
In addition to the aforementioned dependencies, it is also dependent on `lightning`, `BENv2Stats.py` and `BENv2TorchUtils.py`, 
as it sets some default values from these files.