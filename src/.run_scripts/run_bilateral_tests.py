from runs import RunManager
from transforms import TransformFactory
from models import ModelFactory


if __name__ == '__main__':
    # run_datasets = ['bigearthnet', 'imagenet']
    run_models = ModelFactory().all_model_names
    diameters = [3, 7, 13, 0]

    run_datasets = ['bigearthnet']
    diameters = [0]
    run_models = ['resnet']
    # run_models = ['resnet', 'vit', 'efficientnet', 'swin']

    val_transforms = TransformFactory().get_transforms(
        transform_name='bilateral',
        param_name='d',
        param_values=diameters,
    )

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        eval_transforms=val_transforms,
        continue_on_error=False,
        test_run=False,
        device=3,
    )
    run_manager.execute_runs()
