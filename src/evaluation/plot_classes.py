from plotting import ResultsReader
import warnings
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

reader_data = ResultsReader().read_data(
    data_path='C:/CODE/master-thesis/data/results_v7.csv',
    filter_for_transforms='single',
)

upper_bounds = {
    'deepglobe': {'Rangeland': 0.6744, 'Forest': 0.7991, 'Barren': 0.6079, 'Agriculture': 0.9345, 'Water': 0.5881, 'Urban': 0.853, 'macro': 0.7428, 'micro': 0.8215},
    'bigearthnet':   {'Agriculture': 0.8646, 'Agro-forestry': 0.4988, 'Arable': 0.0542, 'Broad-leaved': 0.9028, 'Coastal': 0.625, 'Coniferous': 0.5352, 'Crops': 0.7769, 'Cultivation': 0.446, 'Grassland': 0.9961, 'Industrial Units': 0.9524, 'Marine': 0.8603, 'Mixed forest': 0.481, 'Moors': 0.8426, 'Pastures': 0.307, 'Sand': 0.0441, 'Transitional woodland': 0.4752, 'Urban': 0.8382, 'Waters': 0.7208, 'Wetlands': 0.9493, 'macro': 0.6405, 'micro': 0.8475}
}
lower_bounds = {
    'deepglobe':  {'Rangeland': 0.4127, 'Forest': 0.1761, 'Barren': 0.2164, 'Agriculture': 0.6885, 'Water': 0.1811, 'Urban': 0.3197, 'macro': 0.3324, 'micro': 0.4968},
    'bigearthnet':  {'Agriculture': 0.2385, 'Agro-forestry': 0.0274, 'Arable': 0.0025, 'Broad-leaved': 0.1131, 'Coastal': 0.063, 'Coniferous': 0.0334, 'Crops': 0.2348, 'Cultivation': 0.1278, 'Grassland': 0.1082, 'Industrial Units': 0.5251, 'Marine': 0.1375, 'Mixed forest': 0.1319, 'Moors': 0.296, 'Pastures': 0.0027, 'Sand': 0.0057, 'Transitional woodland': 0.0186, 'Urban': 0.1327, 'Waters': 0.2414, 'Wetlands': 0.3481, 'macro': 0.1468, 'micro': 0.2104}
}

deepglobe_label_mapping = {
    'Urban': 'Urban',
    'Industrial Units': 'Agriculture',
    'Arable': 'Rangeland',
    'Crops': 'Forest',
    'Pastures': 'Water',
    'Cultivation': 'Barren',
}
ben_label_mapping = {
    'Coniferous': 'Coniferous Forest',
    'Marine': 'Marine Waters',
    'Broad-leaved': 'Broad-leaved F.',
    'Waters': 'Inland Waters',
    'Marine': 'Marine Waters',
    'Arable': 'Arable Land',
    'Wetlands': 'Inland Wetlands',
    'Coastal': 'Coastal Wetlands',
    'Transitional woodland': 'Trans. Woodland',
}

# Assuming your DataFrame is named df
max_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()]
min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]
data_values = max_intensity

# Group by dataset and transform, then calculate the average score across models
scores = max_intensity.groupby(['dataset', 'transform'])['score'].mean().reset_index()
cleaned_scores = max_intensity.groupby(['dataset', 'transform'])['cleaned_score'].mean().reset_index()
dataset_scores = cleaned_scores.sort_values(by='transform')


transform_to_category = {
    'Bilateral~Channel Shuffle': 'Shape',
    'Bilateral~Patch Shuffle': 'Spectral',
    'Patch Shuffle~Channel Shuffle': 'Texture'
}

for dataset in ['rgb_bigearthnet']:
    # plot bar chart for 9 subplots
    # fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    # fig.suptitle(f"{dataset} - Cleaned Scores")
    # plot_idx = 0
    dataset_data = data_values[data_values['dataset'] == dataset]
    for transform in dataset_data['transform'].unique():
        data = dataset_data[dataset_data['transform'] == transform]
        all_class_scores = {}
        min_class_scores = {}
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            class_scores = json.loads(model_data['class_scores'].values[0])
            class_scores = {k.split('test_mAP_')[1]: v for k, v in class_scores.items() if 'test_mAP_' in k}
            for class_name, class_score in class_scores.items():
                all_class_scores[class_name] = [class_score] if all_class_scores.get(class_name) is None else all_class_scores[class_name] + [class_score]
        # average class scores
        avg_class_scores = {k: round(sum(v) / len(v), 4) for k, v in all_class_scores.items()}
        # sort by key
        avg_class_scores = dict(sorted(avg_class_scores.items(), key=lambda x: x[0]))
        # rename keys if deepglobe
        if dataset == 'deepglobe':
            avg_class_scores = {deepglobe_label_mapping.get(k, k): v for k, v in avg_class_scores.items()}

        # calculate clean score with upper bounds and lower bounds
        cleaned_scores = {}
        for class_name, class_score in avg_class_scores.items():
            upper_bound = upper_bounds[dataset][class_name]
            lower_bound = lower_bounds[dataset][class_name]
            bound_difference = upper_bound - lower_bound
            if bound_difference == 0:
                cleaned_scores[class_name] = 0
                continue
            cleaned_scores[class_name] = (class_score - lower_bound) / bound_difference
        cleaned_scores = {k: round(v, 4) for k, v in cleaned_scores.items()}
        if dataset == 'bigearthnet' or dataset == 'rgb_bigearthnet':
            cleaned_scores = {ben_label_mapping.get(k, k): v for k, v in cleaned_scores.items()}
        print(transform, '\n', avg_class_scores)
        print(cleaned_scores)

        # plot normal
        plt.figure(figsize=(6, 6))
        # adjust plots to the top
        if dataset == 'deepglobe':
            plt.subplots_adjust(top=0.95, bottom=0.175, left=0.11)
        if dataset == 'bigearthnet' or dataset == 'rgb_bigearthnet':
            plt.subplots_adjust(top=0.95, bottom=0.25, left=0.11)
        cleaned_scores_without_avgs = {k: v for k, v in cleaned_scores.items() if k not in ['macro', 'micro']}
        try:
            cleaned_scores_without_avgs['Trans. Woodland'] = cleaned_scores_without_avgs.pop('Transitional woodland')
        except KeyError:
            pass
        plt.bar(cleaned_scores_without_avgs.keys(), cleaned_scores_without_avgs.values())
        title_name = transform.replace('~', ' & ')
        replace_dict = {'Bilateral & Channel Shuffle': 'Shape Remaining', 'Bilateral & Patch Shuffle': 'Spectral Remaining', 'Patch Shuffle & Channel Shuffle': 'Texture Remaining'}
        for k, v in replace_dict.items():
            title_name = title_name.replace(k, v)
        # plt.title(title_name)
        plt.xlabel('Class')
        plt.ylabel('Relative mAP Score')

        xticks = list(cleaned_scores_without_avgs.keys())
        plt.xticks(list(cleaned_scores_without_avgs.keys()), rotation=90)
        plt.yticks(ticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        # plt.ylim(-0.1, 1)
        plt.ylim(0, 1)
        # plot dashed line for 0
        plt.axhline(0, color='black', linestyle='-')
        # plt.axhline(cleaned_scores['micro'], color='navy', linestyle=':')
        plt.axhline(cleaned_scores['macro'], color='navy', linestyle='--')
        save_path = f"C:/CODE/master-thesis/results/v7/classwise/{dataset}"
        os.makedirs(save_path, exist_ok=True)
        save_file = f"{save_path}/{transform}.png"
        plt.savefig(save_file)
        plt.show()
        plt.close()





