import os
from runs import RunManager
from models import ModelFactory

models = list(ModelFactory().all_models.keys())
datasets = ['deepglobe']

run_manager = RunManager(
    models=models,
    datasets=datasets,
    continue_on_error=False,
    test_run=True,
    verbose=True,
)
run_manager.execute_runs()