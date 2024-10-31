import os
from runs import RunManager
from models import ModelFactory


# models: resnet,efficientnet,convnext,regnet,densenet,resnext,mobilenet,xception,inception,regnety,vit,deit,swin,cait,pvt,pit,beit,convmixer,mvit
models = list(ModelFactory().all_models.keys())
datasets = ['rgb_bigearthnet' 'bigearthnet', 'caltech', 'caltech_120', 'deepglobe']
pretrained = False


# take first argument as dataset
try:
    dataset = os.sys.argv[1]
    datasets = [dataset]
except IndexError:
    print("No input dataset")

try:
    models_str = os.sys.argv[2]
    print(models_str)
    try:
        model_start_index = int(models_str)
        models = models[model_start_index:]
    except:
        models = models_str.split(',')
except IndexError:
    print("No input models")

try:
    pretrained_str = os.sys.argv[3]
    if pretrained_str == 'Pretrained':
        pretrained = True
    else:
        pretrained = False
except IndexError:
    print("No input pretrained")




run_manager = RunManager(
    models=models,
    datasets=datasets,
    continue_on_error=True,
    train=True,
    pretrained=pretrained,
    test_run=False,
    verbose=False,
)
run_manager.execute_runs()