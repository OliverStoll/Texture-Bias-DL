from evaluation.results.reader import ResultsReader
import warnings

from evaluation.latex.constants import dataset_names_friendly, model_names_friendly, metric_per_dataset

warnings.filterwarnings("ignore")


def print_baseline(data_path='/data/run_results.csv'):
    reader = ResultsReader()
    reader_data = reader.read_data(data_path=data_path)
    min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]
    scores = min_intensity.groupby(['dataset', 'model'])['score'].mean().reset_index()
    scores['metric'] = scores['dataset'].apply(lambda x: metric_per_dataset[x])
    scores['dataset'] = scores['dataset'].apply(lambda x: dataset_names_friendly[x])
    scores['model'] = scores['model'].apply(lambda x: model_names_friendly[x])
    scores = scores.rename(
        columns={'dataset': 'Dataset', 'model': 'Model', 'score': 'Test Performance', 'metric': 'Metric'}
    )
    for dataset in ['bigearthnet', 'deepglobe', 'imagenet', 'caltech', 'rgb_bigearthnet', 'caltech_120', 'caltech_ft']:
        dataset_friendly = dataset_names_friendly[dataset]
        dataset_scores = scores[scores['Dataset'] == dataset_friendly]
        dataset_scores = dataset_scores.drop(columns=['Dataset'])
        dataset_scores = dataset_scores.rename(
            columns={'Test Performance': f"{metric_per_dataset[dataset]} Test Performance"}
        )
        output_str = (
            f"\\begin{{table}}[H]\n"
            f"\\centering\n"
            f"\\caption{{Original {metric_per_dataset[dataset]} test performance of "
            f"models on the {dataset_friendly} dataset with no transformations applied.}}\n"
            f"{dataset_scores.to_latex(index=False, float_format='%.3f')}\n"
            f"\\end{{table}}\n"
        )

        print(output_str)

if __name__ == "__main__":
    print_baseline()










