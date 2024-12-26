from plotting import ResultsReader
import warnings
import json
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

reader_data = ResultsReader().read_data(
    data_path='C:/CODE/master-thesis/data/results_v6.csv',
)

upper_bounds = {
    'deepglobe': {'agricultural_land': 0.8228, 'barren_land': 0.8261, 'forest_land': 0.9072, 'rangeland': 0.7045, 'urban_land': 0.8539, 'water': 0.8495},
    'bigearthnet':  {'Agriculture': 0.8962, 'Agro': 0.9765, 'Arable': 0.9978, 'Broad': 0.9637, 'Coastal': 0.9536, 'Coniferous': 0.9716, 'Crops': 0.8717, 'Cultivation': 0.8744, 'Grassland': 0.9936, 'Industrial Units': 0.8599, 'Marine': 0.9396, 'Mixed forest': 0.8762, 'Moors': 0.8577, 'Pastures': 0.9974, 'Sand': 0.9929, 'Transitional woodland': 0.9838, 'Urban': 0.9331, 'Waters': 0.8356, 'Wetlands': 0.9087},
}
lower_bounds = {
    'deepglobe':  {'agricultural_land': 0.6474, 'barren_land': 0.784, 'forest_land': 0.7526, 'rangeland': 0.539, 'urban_land': 0.6098, 'water': 0.812},
    'bigearthnet':  {'Agriculture': 0.7536, 'Agro': 0.801, 'Arable': 0.9979, 'Broad': 0.8811, 'Coastal': 0.918, 'Coniferous': 0.9667, 'Crops': 0.7451, 'Cultivation': 0.7771, 'Grassland': 0.885, 'Industrial Units': 0.494, 'Marine': 0.4119, 'Mixed forest': 0.8681, 'Moors': 0.6776, 'Pastures': 0.9801, 'Sand': 0.9656, 'Transitional woodland': 0.9477, 'Urban': 0.8692, 'Waters': 0.7022, 'Wetlands': 0.6345},
}

# Assuming your DataFrame is named df
max_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()]
min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]
data_values = max_intensity

# Group by dataset and transform, then calculate the average score across models
scores = max_intensity.groupby(['dataset', 'transform'])['score'].mean().reset_index()
cleaned_scores = max_intensity.groupby(['dataset', 'transform'])['cleaned_score'].mean().reset_index()
dataset_scores = cleaned_scores.sort_values(by='transform')


for dataset in ['deepglobe', 'bigearthnet']:
    # plot bar chart for 9 subplots
    ax, fig = plt.subplots(3, 3, figsize=(15, 15))
    plot_idx = 0
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
        # calculate clean score with upper bounds
        cleaned_scores = {}
        for class_name, class_score in avg_class_scores.items():
            cleaned_scores[class_name] = class_score / upper_bounds[dataset][class_name]
        cleaned_scores = {k: round(v, 4) for k, v in cleaned_scores.items()}
        print(transform, '\n', avg_class_scores)
        print(cleaned_scores)




        subplot_ax = fig[plot_idx // 3, plot_idx % 3]
        subplot_ax.bar(cleaned_scores.keys(), cleaned_scores.values())
        subplot_ax.set_title(f"{transform}")
        subplot_ax.set_xlabel('Class')
        subplot_ax.set_ylabel('Score')
        subplot_ax.set_xticks(ticks=list(cleaned_scores.keys()))
        subplot_ax.set_ylim(0.5, 1)
        plot_idx += 1
    plt.show()
print()






