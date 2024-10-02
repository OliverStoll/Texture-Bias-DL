from runs import RunManager


if __name__ == '__main__':
    run_datasets = ['bigearthnet']
    run_models = ['efficientnet', 'swin', 'convnext', 'resnet', 'vit']
    detection_types = ['sobel', 'canny']

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        val_transform_type='edges',
        val_transform_params=detection_types,
        continue_on_error=False,
        train=False,
        test_run=False,
    )
    run_manager.execute_runs()