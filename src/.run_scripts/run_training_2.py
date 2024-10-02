from runs import RunManager
from models import ModelFactory

models = list(ModelFactory().all_models.keys())
datasets = ['bigearthnet']

run_manager = RunManager(
    models=models,
    datasets=datasets,
    device=2,
    test_run=False,
    verbose=True,
    continue_on_error=True,
)
run_manager.execute_runs()