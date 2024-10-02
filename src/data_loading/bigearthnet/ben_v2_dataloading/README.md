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

## Environment
The exact environment used for this project is listed in the `BENDataLoading-requirements.yaml` file as a conda 
requirement specification. To create a conda environment with the same packages, run the following command:

```bash
conda env create -f BENDataLoading-requirements.yaml
```

Alternatively, you can install the following packages:
- python
- python-lmdb
- pandas
- fastparquet (or pyarrow)
- safetensors (huggingface channel)
- pytorch (pytorch channel) (only for the `BENv2DataSet.py` and `BENv2DataModule.py` files and their classes as well as 
the `DM_and_DS_testing.ipynb` notebook)
- lightning (only for the `DM_and_DS_testing.ipynb` notebook where the DataModule is used or if you want to use the 
`BENv2DataModule` class in general)
- jupyterlab (only for the `DM_and_DS_testing.ipynb` notebook)
- ipywidgets (only for the `DM_and_DS_testing.ipynb` notebook)
- jupyter (only for the `DM_and_DS_testing.ipynb` notebook)

## Data

The data used for this project is located in the `/data/kaiclasen/` directory on mars. There is currently no mirror on 
erde or storagecube. See `DM_and_DS_testing.ipynb` for an example of how to load the data.

## Contact

If you have any questions, feel free to contact me (Leo) on slack or via [email](mailto:l.hackel@tu-berlin.de) or visit me in
my office (EN 603).