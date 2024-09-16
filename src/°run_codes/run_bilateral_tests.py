from runs import RunManager
from transforms import TransformFactory


if __name__ == '__main__':
    run_datasets = ['imagenet']
    run_models = ['convnext', 'resnet', 'vit', 'swin', 'efficientnet']
    diameters = [3, 5, 9, 15]

    val_transforms = TransformFactory().get_transforms(
        transform_type='bilateral',
        param_name='d',
        param_values=diameters,
    )

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        val_transforms=val_transforms,
        continue_on_error=True,
        train=False,
        test_run=True,
    )
    run_manager.execute_runs()