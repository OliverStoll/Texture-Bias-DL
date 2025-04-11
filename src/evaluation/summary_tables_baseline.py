from plotting import ResultsReader
import warnings
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")




dataset_names_friendly = {
    'imagenet': '\\as{imnet}',
    'caltech': '\\as{caltech}',
    'caltech_120': '\\as{caltech-120}',
    'caltech_ft': '\\as{caltech-ft}',
    'bigearthnet': '\\as{ben}',
    'rgb_bigearthnet': '\\as{rgb-ben}',
    'deepglobe': '\\as{deepg}',
}


model_names_friendly = {
    'resnet': 'ResNet',
    'efficientnet': 'EfficientNet',
    'convnext': 'ConvNeXt',
    'regnet': 'RegNetX',
    'dense': 'DenseNet',
    'resnext': 'ResNeXt',
    'mobilenet': 'MobileNetV3',
    'xception': 'Xception',
    'inception': 'Inception',
    'densenet': 'DenseNet',
    'regnety': 'RegNetY',
    'deit': 'DeiT',
    'cait': 'CaiT',
    'pvt': 'PVT',
    'vit': 'ViT',
    'pit': 'PiT',
    'beit': 'BeiT',
    'convmixer': 'ConvMixer',
    'mvit': 'MViT',
}

metric_per_dataset = {
    'bigearthnet': 'mAP macro',
    'rgb_bigearthnet': 'mAP macro',
    'deepglobe': 'mAP macro',
    'imagenet': 'Top-1 Acc. macro',
    'caltech': 'Top-1 Acc. macro',
    'caltech_120': 'Top-1 Acc. macro',
    'caltech_ft': 'Top-1 Acc. macro',
}


reader_data = ResultsReader().read_data(
    data_path='/data/run_results.csv',
)
# filter out convnext for caltech and caltech_120
reader_data = reader_data[~((reader_data['dataset'] == 'caltech') & (reader_data['model'] == 'convnext'))]
reader_data = reader_data[~((reader_data['dataset'] == 'caltech_120') & (reader_data['model'] == 'convnext'))]

# Assuming your DataFrame is named df
min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]

# Group by dataset and transform, then calculate the average score across models
scores = min_intensity.groupby(['dataset', 'model'])['score'].mean().reset_index()
# replace dataset names with friendly names
# scores['metric'] = scores['dataset'].apply(lambda x: metric_per_dataset[x])
scores['dataset'] = scores['dataset'].apply(lambda x: dataset_names_friendly[x])
scores['model'] = scores['model'].apply(lambda x: model_names_friendly[x])
scores = scores.rename(columns={'dataset': 'Dataset', 'model': 'Model', 'score': 'Test Performance', 'metric': 'Metric'})


for dataset in ['bigearthnet', 'deepglobe', 'imagenet', 'caltech', 'rgb_bigearthnet', 'caltech_120', 'caltech_ft']:
    dataset_friendly = dataset_names_friendly[dataset]
    dataset_scores = scores[scores['Dataset'] == dataset_friendly]
    dataset_scores = dataset_scores.drop(columns=['Dataset'])
    # rename Test Performance to the metric for the dataset
    dataset_scores = dataset_scores.rename(columns={'Test Performance': f"{metric_per_dataset[dataset]} Test Performance"})
    print('\n\n\n\\begin{table}[H]')
    print('\\centering')
    print('\\caption{Original ' + metric_per_dataset[
        dataset] + ' test performance of models on the ' + dataset_friendly + ' dataset with no transformations applied.}')
    print(dataset_scores.to_latex(index=False, float_format="%.3f"), end='')
    print('\\end{table}\n')










