from run_single import RunManager


if __name__ == '__main__':
    run_datasets = ['bigearthnet']
    run_models = ['efficientnet', 'swin', 'convnext', 'resnet', 'vit']
    cutoffs = [100000]

    run_manager = RunManager(
        models=run_models,
        datasets=run_datasets,
        val_transform_type='low_pass',
        val_transform_params=cutoffs,
        continue_on_error=False,
        train=False,
        test_run=False,
    )
    run_manager.execute_multiple_runs()
    # TODO: find why the low-pass filter is crippling mAP even with high cutoffs
    # TODO: run when BEN is fully trained