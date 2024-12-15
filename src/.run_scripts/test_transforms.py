import os
from run_manager import RunManager
from transforms.transforms_factory import TransformFactory
from models import ModelFactory
from data_loading.datasets import DataLoaderFactory


run_datasets = DataLoaderFactory().dataset_names
run_models = ModelFactory().all_model_names
val_transforms = TransformFactory().get_all_default_transforms()
# get only noise
# val_transforms = TransformFactory().get_single_default_transform('noise')


# take first argument as dataset
try:
    dataset = os.sys.argv[1]
    run_datasets = [dataset]
except IndexError:
    print("No input dataset")

try:
    models_str = os.sys.argv[2]
    print(models_str)
    try:
        model_start_index = int(models_str)
        run_models = run_models[model_start_index:]
    except:
        run_models = models_str.split(',')
except IndexError:
    print("No input models")


run_manager = RunManager(
    models=run_models,
    datasets=run_datasets,
    eval_transforms=val_transforms,
    continue_on_error=True,
    test_run=100,
    train=False,
)
run_manager.execute_multiple_runs()
