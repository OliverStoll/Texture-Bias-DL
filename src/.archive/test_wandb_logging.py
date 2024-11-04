from run_single import SingleRun


run = SingleRun(
    dataset_name='bigearthnet',
    model_name='resnet',
    devices=[2],
)
run.execute()