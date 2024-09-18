from runs import RunManager
from models import ModelCollection

models = list(ModelCollection().model_dict.keys())
datasets = ['bigearthnet', 'imagenet']

run_manager = RunManager(
    models=models,
    datasets=datasets,
    continue_on_error=True,
    test_run=False,
    device=3,
)
run_manager.execute_runs()