import os
import itertools
import warnings
import pandas as pd
from common_utils.logger import create_logger

from src.evaluation.plots.plotting import ResultsPlotter


class PlottingManager:
    """
    Orchestrates the generation of plots from experimental results using different transformation settings.
    """

    log = create_logger("Plot Pipeline")
    output_path = 'C:/CODE/master-thesis/results'
    score_types = ['cleaned_score', 'relative_loss', 'score', 'absolute_loss', 'relative_score']


    def __init__(
            self,
            data_path: str | None = None,
            output_path: str | None = None
    ) -> None:
        """
        Initialize the plotting manager.

        Args:
            data_path: Optional path to the input results file.
            output_path: Optional directory to store output plots.
        """
        self.data_path = data_path
        self.output_path = output_path or self.output_path
        os.makedirs(self.output_path, exist_ok=True)
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        self.plot_functions = {
            'single': self.plot_single_transforms,
            'paired': self.plot_paired_transforms,
        }


    def plot_all(
            self,
            score_types: list[str] | None = None,
            exclude_plot_type: str | None = None
    ) -> None:
        """
        Generate and save all plots for all score types and configurations.

        Args:
            score_types: List of score types to plot. If None, all predefined types are used.
            exclude_plot_type: Optional plot type to skip (e.g., 'paired' or 'single').
        """
        score_types = score_types or self.score_types
        self.log.info(f"Plotting all results to {self.output_path}")

        for score_type, (plot_type, plot_function), split_by_architecture, use_subplots in itertools.product(
                score_types,
                self.plot_functions.items(),
                [True, False],
                [True, False],
        ):
            if plot_type == exclude_plot_type:
                continue

            output_dir = (
                f"{self.output_path}/{score_type}/{plot_type}"
                f"{'/split-architecture' if split_by_architecture else '/default'}"
            )
            os.makedirs(output_dir, exist_ok=True)

            self.log.info(f"\nPlotting {score_type} [{plot_type} | {split_by_architecture}]\n")

            plot_function(  # type: ignore
                score_type=score_type,
                output_dir=output_dir,
                plot_split_by_model_type=split_by_architecture,
            )

    def plot_single_transforms(
            self,
            output_dir: str,
            score_type: str,
            plot_split_by_model_type: bool,
    ) -> None:
        """
        Plot results for all single transformations.

        Args:
            output_dir: Directory to store the generated plots.
            score_type: Score type to be visualized (e.g., 'relative_loss').
            plot_split_by_model_type: Whether to separate plots by model architecture type.
        """
        self.log.info(f"Plotting single transforms [{score_type} | {plot_split_by_model_type}]")

        plotter = ResultsPlotter(
            data_path=self.data_path,
            output_dir=output_dir,
            filter_for_transforms='single',
            plot_split_by_model_type=plot_split_by_model_type,
            plot_as_subplots=False,
            score_type=score_type,
        )

        self._plot_dataset_categories_single(plotter, output_dir)


    def plot_paired_transforms(
            self,
            output_dir: str,
            score_type: str = 'relative_loss',
            transform_combis: list[str] | None = None,
            plot_split_by_model_type: bool = False,
    ) -> None:
        """
        Plot results for paired (combined) transformations.

        Args:
            output_dir: Directory to store the generated plots.
            score_type: Score type to be visualized (default is 'relative_loss').
            transform_combis: Optional list of specific transformation combinations to plot.
            plot_split_by_model_type: Whether to separate plots by model architecture type.
        """
        self.log.info(f"Plotting double transforms [{score_type} | {plot_split_by_model_type}]")

        plotter = ResultsPlotter(
            data_path=self.data_path,
            filter_for_transforms='combined',
            output_dir=output_dir,
            plot_as_subplots=False,
            score_type=score_type,
            plot_split_by_model_type=plot_split_by_model_type,
        )

        transform_combis = transform_combis or self._get_transform_pairs()

        plotter.create_all_plots(
            save_name='',
            transform_names=transform_combis
        )

        self._plot_dataset_categories_paired(
            plotter=plotter,
            output_dir=output_dir,
            transform_combis=transform_combis
        )


    @staticmethod
    def _plot_dataset_categories_single(
            plotter: ResultsPlotter,
            output_dir: str
    ) -> None:
        """
        Generate plots for all dataset categories using single transformations.

        Args:
            plotter: An instance of ResultsPlotter configured for single transformations.
            output_dir: Directory where plots will be saved.
        """
        for dataset_category in ['RS', 'CV', 'BEN', 'CAL_FT', 'ALL']:
            os.environ['Y_LABEL'] = 'Macro Acc. ' if dataset_category == 'CV' else 'Macro mAP '
            category_output_dir = f"{output_dir}/{dataset_category}"
            os.makedirs(category_output_dir, exist_ok=True)

            dataset_names = plotter.dataset_categories[dataset_category]

            # Plot all transforms for this dataset category
            plotter.create_all_plots(
                save_name=f"{dataset_category}",
                dataset_names=dataset_names,
            )

            # Plot each transform category separately
            for transform_category, transform_names in plotter.transform_categories.items():
                plotter.create_all_plots(
                    save_name=f"{dataset_category}",
                    transform_names=transform_names,
                    dataset_names=dataset_names,
                )


    @staticmethod
    def _plot_dataset_categories_paired(
            plotter: ResultsPlotter,
            output_dir: str,
            transform_combis: list[str]
    ) -> None:
        """
        Generate plots for all dataset categories using paired (combined) transformations.

        Args:
            plotter: An instance of ResultsPlotter configured for paired transformations.
            output_dir: Directory where plots will be saved.
            transform_combis: List of transformation combinations to plot.
        """
        for dataset_category in ['RS', 'CV', 'BEN', 'CAL_FT', 'ALL']:
            os.environ['Y_LABEL'] = 'Macro Acc. ' if dataset_category == 'CV' else 'Macro mAP '
            category_output_dir = f"{output_dir}/{dataset_category}"
            os.makedirs(category_output_dir, exist_ok=True)

            dataset_names = plotter.dataset_categories[dataset_category]

            plotter.create_all_plots(
                save_name=dataset_category,
                transform_names=transform_combis,
                dataset_names=dataset_names,
            )


    @staticmethod
    def _get_transform_pairs() -> list[str]:
        """
        Define and return all relevant pairs of transformations to evaluate.

        Returns:
            A list of transformation name pairs joined by '~'.
        """
        transform_pairs = []
        for texture_transform, shape_transform, color_transform in itertools.product(
                ['bilateral'],
                ['patch_shuffle'],
                ['channel_shuffle'],
        ):
            transform_pairs.append(f"{texture_transform}~{shape_transform}")
            transform_pairs.append(f"{texture_transform}~{color_transform}")
            transform_pairs.append(f"{shape_transform}~{color_transform}")
        return transform_pairs



if __name__ == '__main__':
    from warnings import filterwarnings

    filterwarnings('ignore', category=UserWarning)

    plotter_ = PlottingManager(
        data_path=f'C:/CODE/master-thesis/data/run_results.csv',
        output_path=f'C:/CODE/master-thesis/results/final',
    )
    plotter_.plot_all(
        score_types=['cleaned_score'],
    )