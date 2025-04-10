from plotting import ResultsReader
import warnings
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")




dataset_names_friendly = {
    'imagenet': 'ImageNet',
    'caltech': 'Caltech',
    'caltech_120': 'Caltech 120',
    'caltech_ft': 'Caltech Finet.',
    'bigearthnet': 'BigEarthNet',
    'rgb_bigearthnet': 'RGB BigEarthNet',
    'deepglobe': 'DeepGlobe',
}



reader_data = ResultsReader().read_data(
    data_path='/data/run_results.csv',
)
# filter out convnext for caltech and caltech_120
reader_data = reader_data[~((reader_data['dataset'] == 'caltech') & (reader_data['model'] == 'convnext'))]
reader_data = reader_data[~((reader_data['dataset'] == 'caltech_120') & (reader_data['model'] == 'convnext'))]

# Assuming your DataFrame is named df
max_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()]
min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]

# merge score of min_intensity with max_intensity
max_intensity = max_intensity.merge(min_intensity[['dataset', 'model', 'transform', 'score']], on=['dataset', 'model', 'transform'], suffixes=('', '_baseline'))
data = max_intensity

# Group by dataset and transform, then calculate the average score across models
scores = max_intensity.groupby(['dataset', 'transform', 'model_type'])['score'].mean().reset_index()
scores_baseline = max_intensity.groupby(['dataset', 'transform', 'model_type'])['score_baseline'].mean().reset_index()
cleaned_scores = max_intensity.groupby(['dataset', 'transform', 'model_type'])['cleaned_score'].mean().reset_index()

# merge cleaned scores with scores
cleaned_scores = cleaned_scores.merge(scores, on=['dataset', 'transform', 'model_type'])
cleaned_scores = cleaned_scores.merge(scores_baseline, on=['dataset', 'transform', 'model_type'], suffixes=('', '_baseline'))
# merge

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
overall_data = data.sort_values(by=['domain'], ascending=False)

for transforms in [['Channel Shuffle', 'Channel Inversion', 'Channel Mean'], ['Bilateral', 'Median', 'Gaussian'], ['Patch Shuffle', 'Patch Rotation']]:
    for model_type in ['transformer', 'cnn']:
        data = overall_data[overall_data['Model'].str.contains(model_type)]
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









