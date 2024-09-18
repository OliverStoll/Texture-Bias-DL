from runs import RunManager
from models import ModelCollection

models = list(ModelCollection().all_models.keys())
datasets = ['bigearthnet', 'imagenet']

run_manager = RunManager(
    models=models,
    datasets=datasets,
    device=3,
    test_run=False,
    verbose=True,
)
run_manager.execute_runs()