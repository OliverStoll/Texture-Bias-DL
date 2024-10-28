from runs import RunManager
from transforms import TransformFactory


if __name__ == '__main__':
    run_datasets = ['imagenet']
    run_models = ['convnext', 'resnet', 'vit', 'swin']  # TODO: run when BEN is fully trained
    grid_sizes = range(1, 21)
    grid_sizes = [2]

    grid_shuffle_transforms = TransformFactory().get_transforms(
        transform_name='grid_shuffle',
        param_name='grid_size',
        param_values=grid_sizes,
    )

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        eval_transforms=grid_shuffle_transforms,
        continue_on_error=True,
        train=False,
        test_run=True,
    )
    run_manager.execute_runs()