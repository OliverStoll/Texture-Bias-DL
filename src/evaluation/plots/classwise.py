from evaluation.plots.constants import ben_label_mapping, deepglobe_label_mapping, lower_bounds, upper_bounds
from evaluation.results.reader import ResultsReader
import warnings
import json
import os
import matplotlib.pyplot as plt
import pandas as pd


class ClasswisePlotter:
    """
    Generates class-wise performance plots highlighting feature importance of individual classes for their prediction.

    Attributes:
        datasets: List of dataset names to process.
        transform_friendly_names: Mapping of transform combinations to readable names.
        layout_adjustments: Custom subplot layout settings per dataset.
        y_ticks: Y-axis tick values for plotting class-wise metrics.
    """

    datasets = ['bigearthnet', 'deepglobe']

    transform_friendly_names = {
        'Bilateral & Channel Shuffle': 'Shape Remaining',
        'Bilateral & Patch Shuffle': 'Spectral Remaining',
        'Patch Shuffle & Channel Shuffle': 'Texture Remaining'
    }

    layout_adjustments = {
        'deepglobe': {'top': 0.95, 'bottom': 0.175, 'left': 0.11},
        'bigearthnet': {'top': 0.95, 'bottom': 0.25, 'left': 0.11},
        'rgb_bigearthnet': {'top': 0.95, 'bottom': 0.25, 'left': 0.11},
    }

    y_ticks = [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    def __init__(
            self,
            data_path: str,
            save_path: str,
            datasets: list[str] | None = None,
            transform_type: str = 'single'
    ) -> None:
        """
        Initialize the ClasswisePlotter.

        Args:
            data_path: Path to the input CSV file with results.
            save_path: Directory where plots will be saved.
            datasets: Optional list of datasets to include. Defaults to class-level datasets.
            transform_type: Type of transformation to filter for ('single' or 'combined').
        """
        self.datasets = datasets or self.datasets
        self.reader = ResultsReader()
        self.data_path = data_path
        self.save_path = save_path
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        raw_data = self.reader.read_data(
            data_path=self.data_path,
            filter_for_transforms=transform_type,
        )
        self.results_data = self._prepare_data(raw_data)

    def _prepare_data(self, reader_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the rows corresponding to the maximum transformation intensity
        per (dataset, model, transform) combination.

        Args:
            reader_data: Raw results DataFrame from the ResultsReader.

        Returns:
            A filtered DataFrame containing only the maximum-intensity rows.
        """
        max_intensity = reader_data.loc[
            reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()
        ]
        return max_intensity

    def plot_all_datasets(self) -> None:
        """
        Generate class-wise plots for each dataset and each applied transformation.
        """
        for dataset in self.datasets:
            dataset_subset = self.results_data[self.results_data['dataset'] == dataset]

            for transform in dataset_subset['transform'].unique():
                transform_subset = dataset_subset[dataset_subset['transform'] == transform]
                cleaned_scores = self.get_classes_transform_score(dataset, transform_subset)

                self.plot_classes_single_dataset_plot(
                    class_scores=cleaned_scores,
                    dataset_name=dataset,
                    transform_name=transform
                )


    def get_classes_transform_score(
            self,
            dataset: str,
            transform_data: pd.DataFrame
    ) -> dict[str, float]:
        """
        Normalize per-class scores for a given dataset and transformation using dataset-specific bounds.

        Args:
            dataset: Name of the dataset (e.g., 'bigearthnet', 'deepglobe').
            transform_data: DataFrame filtered for the specific dataset and transformation.

        Returns:
            A dictionary mapping class names to cleaned (normalized) scores in [0, 1].
        """
        avg_class_scores = self._get_avg_class_score(dataset, transform_data)
        cleaned_scores = {}

        for class_name, class_score in avg_class_scores.items():
            upper_bound = upper_bounds[dataset][class_name]
            lower_bound = lower_bounds[dataset][class_name]
            bound_range = upper_bound - lower_bound

            if bound_range == 0:
                cleaned_scores[class_name] = 0.0
                continue

            normalized_score = (class_score - lower_bound) / bound_range
            cleaned_scores[class_name] = round(normalized_score, 4)

        if dataset in ['bigearthnet', 'rgb_bigearthnet']:
            cleaned_scores = {
                ben_label_mapping.get(class_name, class_name): score
                for class_name, score in cleaned_scores.items()
            }

        return cleaned_scores


    @staticmethod
    def _get_avg_class_score(
            dataset: str,
            transform_data: pd.DataFrame
    ) -> dict[str, float]:
        """
        Compute the average class-wise score across all models for a given dataset and transformation.

        Args:
            dataset: Name of the dataset (e.g., 'deepglobe', 'bigearthnet').
            transform_data: DataFrame containing model-specific results and class scores.

        Returns:
            A dictionary mapping class names to their average scores, sorted alphabetically.
        """
        all_class_scores = {}

        for model_name in transform_data['model'].unique():
            model_data = transform_data[transform_data['model'] == model_name]
            class_scores_json = model_data['class_scores'].values[0]
            class_scores = json.loads(class_scores_json)
            # Extract only 'test_mAP_<class>' keys
            class_scores = {
                k.split('test_mAP_')[1]: v
                for k, v in class_scores.items()
                if 'test_mAP_' in k
            }
            for class_name, score in class_scores.items():
                all_class_scores.setdefault(class_name, []).append(score)

        avg_class_scores = {
            class_name: round(sum(scores) / len(scores), 4)
            for class_name, scores in all_class_scores.items()
        }
        avg_class_scores = dict(sorted(avg_class_scores.items(), key=lambda x: x[0]))

        if dataset == 'deepglobe':
            avg_class_scores = {
                deepglobe_label_mapping.get(class_name, class_name): score
                for class_name, score in avg_class_scores.items()
            }

        return avg_class_scores


    def plot_classes_single_dataset_plot(
            self,
            class_scores: dict[str, float],
            dataset_name: str,
            transform_name: str
    ) -> None:
        """
        Plot a single bar chart of class-wise scores for a dataset under a specific transformation.

        Args:
            class_scores: Dictionary mapping class names to normalized scores.
            dataset_name: Name of the dataset (e.g., 'bigearthnet', 'deepglobe').
            transform_name: Name of the transformation applied (e.g., 'bilateral~patch_shuffle').
        """
        plt.figure(figsize=(6, 6))

        if dataset_name in self.layout_adjustments:
            plt.subplots_adjust(**self.layout_adjustments[dataset_name])

        # Remove 'macro' and 'micro' average scores from the bars
        class_scores_filtered = {
            k: v for k, v in class_scores.items() if k not in ['macro', 'micro']
        }

        plt.bar(class_scores_filtered.keys(), class_scores_filtered.values())  # noqa

        # Format and localize title
        title_name = transform_name.replace('~', ' & ')
        for key, friendly in self.transform_friendly_names.items():
            title_name = title_name.replace(key, friendly)

        plt.title(title_name)
        plt.xlabel('Class')
        plt.ylabel('Relative mAP Score')
        plt.xticks(list(class_scores_filtered.keys()), rotation=90)
        plt.yticks(self.y_ticks)
        plt.ylim(-0.1, 1)

        # Horizontal reference lines
        plt.axhline(0, color='black', linestyle='-')
        if 'macro' in class_scores:
            plt.axhline(class_scores['macro'], color='navy', linestyle='--')

        os.makedirs(self.save_path, exist_ok=True)
        save_file = f"{self.save_path}/{transform_name}.png"
        plt.savefig(save_file)
        plt.show()
        plt.close()



if __name__ == '__main__':
    """ Example usage """
    data_path_ = 'C:/CODE/master-thesis/data/run_results.csv'
    save_path_ = 'C:/CODE/master-thesis/data/plots/classwise_TEST'
    plotter = ClasswisePlotter(data_path=data_path_, save_path=save_path_)
    plotter.plot_all_datasets()