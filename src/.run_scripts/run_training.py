import os
from runs import RunManager
from models import ModelFactory


models = list(ModelFactory().all_models.keys())
datasets = ['bigearthnet']

# take first argument as dataset
try:
    dataset = os.sys.argv[1]
    datasets = [dataset]
except IndexError:
    print("No input dataset")



run_manager = RunManager(
    models=models,
    datasets=datasets,
    continue_on_error=False,
    train=True,
    pretrained=False,
    test_run=False,
    verbose=True,
)
run_manager.execute_runs()