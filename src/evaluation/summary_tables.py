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
    'regnetx': 'RegNetX',
    'dense': 'DenseNet',
    'renext': 'ResNeXt',
    'mobilenet': 'MobileNetV3',
    'xception': 'Xception',
    'inception': 'Inception',
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




reader_data = ResultsReader().read_data(
    data_path='C:/CODE/master-thesis/data/results_v7.csv',
)
# filter out convnext for caltech and caltech_120
reader_data = reader_data[~((reader_data['dataset'] == 'caltech') & (reader_data['model'] == 'convnext'))]
reader_data = reader_data[~((reader_data['dataset'] == 'caltech_120') & (reader_data['model'] == 'convnext'))]

# Assuming your DataFrame is named df
max_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()]
# min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]
data = max_intensity

# Group by dataset and transform, then calculate the average score across models
scores = max_intensity.groupby(['dataset', 'transform'])['score'].mean().reset_index()
cleaned_scores = max_intensity.groupby(['dataset', 'transform'])['cleaned_score'].mean().reset_index()
dataset_scores = cleaned_scores.sort_values(by='transform')

best_model = data.loc[data.groupby(["dataset", "transform"])["cleaned_score"].idxmax()][['dataset', 'transform', 'model', 'cleaned_score', 'score']]
avg_model = data.groupby(["dataset", "transform"], as_index=False)["cleaned_score"].mean()

# replace cleaned_score with best_score and avg_score
best_model = best_model.rename(columns={"cleaned_score": "Best Relative Performance"})
avg_model = avg_model.rename(columns={"cleaned_score": "Avg. Relative Performance"})
data = best_model.merge(avg_model, on=["dataset", "transform"])
data = data.rename(columns={"dataset": "Dataset", "transform": "Transformation", "model": "Model"})
# add domain column either cv or rs
rs_datasets = ['BigEarthNet', 'DeepGlobe', 'RGB BigEarthNet']
data['Dataset'] = data['Dataset'].apply(lambda x: dataset_names_friendly[x])
data['domain'] = data['Dataset'].apply(lambda x: 'rs' if x in rs_datasets else 'cv')
# sort by dataset domain
data = data.sort_values(by=['domain'], ascending=False)

for transforms in [['Channel Shuffle', 'Channel Inversion', 'Channel Mean'], ['Bilateral', 'Median', 'Gaussian'], ['Patch Shuffle', 'Patch Rotation']]:
    all_best = []
    all_avg = []
    for dataset in data['Dataset'].unique():
        d_data = data[data['Dataset'] == dataset]
        dataset_best = {}
        dataset_avgs = {}
        for transform in transforms:
            t_data = d_data[d_data['Transformation'] == transform]
            dataset_best[transform] = t_data['Best Relative Performance'].values[0]
            dataset_avgs[transform] = t_data['Avg. Relative Performance'].values[0]
        all_best.append(dataset_best)
        all_avg.append(dataset_avgs)
    transform_best = pd.DataFrame(all_best, index=data['Dataset'].unique())
    transform_best['Average'] = transform_best.mean(axis=1)
    transform_avg = pd.DataFrame(all_avg, index=data['Dataset'].unique())
    transform_avg['Average'] = transform_avg.mean(axis=1)
    # make index a column
    transform_avg = transform_avg.reset_index().rename(columns={"index": "Dataset"})
    transform_best = transform_best.reset_index().rename(columns={"index": "Dataset"})



    print(transform_avg.to_latex(index=False, float_format="%.2f"))
    print(transform_best.to_latex(index=False, float_format="%.2f"))









