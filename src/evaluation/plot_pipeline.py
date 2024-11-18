import os
import itertools
from common_utils.logger import create_logger

from src.evaluation.plotting import ResultsPlotter


class PlotPipeline:
    log = create_logger("Plot Pipeline")
    output_path = 'C:/CODE/master-thesis/results'
    score_types = ['relative_loss', 'score', 'absolute_loss']

    def __init__(self, data_path: str | None = None, output_path: str | None = None):
        self.data_path = data_path
        self.output_path = output_path or self.output_path
        self.plot_functions = {
            'single': self.plot_single_transforms,
            'double': self.plot_double_transforms,
        }

    def plot_all(self, score_types=None):
        score_types = score_types or self.score_types
        self.log.info(f"Plotting all results to {self.output_path}")
        for score_type, plot_item, split_architecture in itertools.product(
                score_types,
                self.plot_functions.items(),
                [True, False]
        ):
            plot_type, plot_function = plot_item
            output_dir = (f"{self.output_path}/{score_type}/{plot_type}"
                          f"{'/show_architecture' if split_architecture else '/merged'}")
            os.makedirs(output_dir, exist_ok=True)
            self.log.info(f"\nPlotting {score_type} [{plot_type} | {split_architecture}]\n")
            plot_function(  # type: ignore
                score_type=score_type,
                output_dir=output_dir,
                plot_split_by_model_type=split_architecture,
            )

    def plot_single_transforms(
            self,
            output_dir: str,
            score_type: str = 'relative_loss',
            plot_split_by_model_type: bool = False,
    ):
        self.log.info(f"Plotting single transforms [{score_type} | {plot_split_by_model_type}]")
        plotter = ResultsPlotter(
            data_path=self.data_path,
            output_dir=output_dir,
            filter_for_transforms='single',
            plot_split_by_model_type=plot_split_by_model_type,
            score_type=score_type,
        )
        self._plot_transform_categories_single(plotter)
        self._plot_dataset_categories_single(plotter, output_dir)

    def plot_double_transforms(
            self,
            output_dir: str,
            score_type: str = 'relative_loss',
            transform_combis: list[str] | None = None,
            plot_split_by_model_type: bool = False,
    ):
        self.log.info(f"Plotting double transforms [{score_type} | {plot_split_by_model_type}]")
        plotter = ResultsPlotter(
            data_path=self.data_path,
            filter_for_transforms='combined',
            output_dir=output_dir,
            plot_as_subplots=False,
            score_type=score_type,
            plot_split_by_model_type=plot_split_by_model_type,
        )
        transform_combis = transform_combis or self._get_default_combinations()
        plotter.create_all_plots(save_name=f'ALL_TRANSFORMS', transform_names=transform_combis)


    @staticmethod
    def _plot_dataset_categories_single(plotter, output_dir):
        for dataset_category in ['RS', 'CV']:
            os.makedirs(f"{output_dir}/{dataset_category}", exist_ok=True)
            dataset_names = plotter.dataset_categories[dataset_category]
            plotter.create_all_plots(
                save_name=f"{dataset_category}/ALL_TRANSFORMS", dataset_names=dataset_names,
            )
            for transform_category, transform_names in plotter.transform_categories.items():
                plotter.create_all_plots(
                    save_name=f"{dataset_category}/{transform_category.upper()}",
                    transform_names=transform_names,
                    dataset_names=dataset_names,
                )

    @staticmethod
    def _plot_transform_categories_single(plotter):
        plotter.create_all_plots(save_name='ALL_RESULTS')
        for transform_category, transform_names in plotter.transform_categories.items():
            plotter.create_all_plots(
                save_name=transform_category.upper(), transform_names=transform_names,
            )

    @staticmethod
    def _get_default_combinations():
        transform_combis_ = []
        for texture_t, shape_t, color_t in itertools.product(
                ['bilateral', 'median', 'gaussian'],
                ['patch_shuffle', 'patch_rotation'],
                ['channel_shuffle', 'channel_inversion', 'greyscale'],
        ):
            transform_combis_.append(f"{texture_t}~{shape_t}")
            transform_combis_.append(f"{texture_t}~{color_t}")
            transform_combis_.append(f"{shape_t}~{color_t}")
        return transform_combis_




if __name__ == '__main__':
    for version in ['v3', 'v2', 'v1']:
        plotter_ = PlotPipeline(
            data_path=f'C:/CODE/master-thesis/data/results_{version}.csv',
            output_path=f'C:/CODE/master-thesis/results/{version}',
        )
        plotter_.plot_all(

            # score_types=['relative_loss']
        )
