from plotting import ResultsReader
import warnings
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation.latex.constants import dataset_names_friendly, model_names_friendly, metric_per_dataset
from evaluation.results.reader import ResultsReader
from evaluation.results.data import prepare_data

warnings.filterwarnings("ignore")


reader_data = ResultsReader().read_data(data_path='/data/run_results.csv')
results_data = prepare_data(reader_data)

for transforms in [['Channel Shuffle', 'Channel Inversion', 'Channel Mean'], ['Bilateral', 'Median', 'Gaussian'], ['Patch Shuffle', 'Patch Rotation']]:
    for model_type in ['transformer', 'cnn']:
        results = results_data[results_data['Model'].str.contains(model_type)]
        all_best = []
        all_avg = []
        for dataset in results['Dataset'].unique():
            d_data = results[results['Dataset'] == dataset]
            dataset_best = {}
            dataset_avgs = {}
            for transform in transforms:
                t_data = d_data[d_data['Transformation'] == transform]
                dataset_best[transform] = t_data['Best Relative Performance'].values[0]
                dataset_avgs[transform] = t_data['Avg. Relative Performance'].values[0]
            all_best.append(dataset_best)
            all_avg.append(dataset_avgs)
        transform_best = pd.DataFrame(all_best, index=results['Dataset'].unique())
        transform_best['Average'] = transform_best.mean(axis=1)
        transform_avg = pd.DataFrame(all_avg, index=results['Dataset'].unique())
        transform_avg['Average'] = transform_avg.mean(axis=1)
        # make index a column
        transform_avg = transform_avg.reset_index().rename(columns={"index": "Dataset"})
        transform_best = transform_best.reset_index().rename(columns={"index": "Dataset"})
        print(transform_avg.to_latex(index=False, float_format="%.2f"))
        print(transform_best.to_latex(index=False, float_format="%.2f"))









