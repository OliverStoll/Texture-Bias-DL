from runs import RunManager
from transforms import TransformFactory
from models import ModelFactory
from datasets import DataLoaderFactory


if __name__ == '__main__':
    run_datasets = ['bigearthnet', 'imagenet']
    run_models = ModelFactory().all_model_names
    val_transforms = TransformFactory().get_all_default_transforms()

    # custom values
    run_models = ['resnet', 'mvit', 'densenet', 'deit']
    val_transforms = TransformFactory().get_transforms(transform_name='color_jitter', param_values=[0])

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        eval_transforms=val_transforms,
        continue_on_error=True,
        test_run=100,
        train=False,
        device=2,
    )
    run_manager.execute_runs()
