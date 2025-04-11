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
    data_path='C:/CODE/master-thesis/data/results_v7.csv',
    filter_for_transforms='combined',
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

all_best = []
all_avg = []
for dataset in data['Dataset'].unique():

    dataset_best = {'Dataset': dataset}
    dataset_avgs = {'Dataset': dataset}
    d_data = data[data['Dataset'] == dataset]
    for transforms in ['Bilateral~Channel Shuffle', 'Bilateral~Patch Shuffle', 'Patch Shuffle~Channel Shuffle']:
        t_data = d_data[d_data['Transformation'] == transforms]
        dataset_best[transforms] = t_data['Best Relative Performance'].values[0]
        dataset_avgs[transforms] = t_data['Avg. Relative Performance'].values[0]
    all_best.append(dataset_best)
    all_avg.append(dataset_avgs)
best_table = pd.DataFrame(all_best)
avg_table = pd.DataFrame(all_avg)
# make Dataset the index
best_table = best_table.set_index('Dataset')
avg_table = avg_table.set_index('Dataset')
# best_table['Average'] = best_table.mean(axis=1)
# avg_table['Average'] = avg_table.mean(axis=1)
best_table['Sum'] = best_table.sum(axis=1)
avg_table['Sum'] = avg_table.sum(axis=1)
# make index a column
avg_table = avg_table.reset_index().rename(columns={"index": "Dataset"})
best_table = best_table.reset_index().rename(columns={"index": "Dataset"})


transform_to_category = {
    'Bilateral~Channel Shuffle': 'Shape',
    'Bilateral~Patch Shuffle': 'Spectral',
    'Patch Shuffle~Channel Shuffle': 'Texture'
}

# rename columns
avg_table = avg_table.rename(columns=transform_to_category)
best_table = best_table.rename(columns=transform_to_category)

# reorder columns: dataset, spectral, texture, shape, sum
avg_table = avg_table[['Dataset', 'Spectral', 'Texture', 'Shape', 'Sum']]
best_table = best_table[['Dataset', 'Spectral', 'Texture', 'Shape', 'Sum']]


print(avg_table.to_latex(index=False, float_format="%.2f"))
print(best_table.to_latex(index=False, float_format="%.2f"))


"""
datasets = t_data['Dataset']
x_vals = np.arange(len(datasets))
width = 0.4
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(f"{transform} - Best vs Average Scores")
ax.bar(x_vals - width / 2, t_data['best_score'], width, color='r', label='Best Score')
# ax.bar(x_vals + width / 2, t_data['avg_score'], width, color='blue', label='Average Score')
ax.set_xticks(x_vals)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.set_ylabel('Relative Performance')
ax.set_xlabel('Dataset')
ax.legend()
fig.tight_layout()
plt.show()
"""








