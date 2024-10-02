from runs import RunManager
from models import ModelFactory

models = list(ModelFactory().all_models.keys())
datasets = ['bigearthnet', 'imagenet']

run_manager = RunManager(
    models=models,
    datasets=datasets,
    device=3,
    test_run=False,
    verbose=True,
)
run_manager.execute_runs()