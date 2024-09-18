from runs import Run


run = Run(
    dataset_name='bigearthnet',
    model_name='resnet',
    devices=[2],
)
run.execute()