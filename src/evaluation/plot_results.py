import itertools
import pandas as pd
import matplotlib.pyplot as plt
from common_utils.config import CONFIG
from common_utils.logger import create_logger


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = f"{CONFIG['output_dir']}/results"
    results_df_path = f"{output_dir}/results.csv"
    all_transform_names = [
        'bilateral', 'median', 'gaussian', 'patch_shuffle',
        'patch_rotation', 'channel_shuffle', 'channel_inversion', 'greyscale'
    ]
    all_dataset_names = [
        'bigearthnet', 'rgb_bigearthnet', 'deepglobe', 'imagenet', 'caltech', 'caltech_120'
    ]
    models = {
        'cnn': ['resnet', 'efficientnet', 'convnext', 'regnet', 'densenet', 'resnext',
                  'mobilenet', 'xception', 'inception', 'regnety'],
        'transformer': ['vit', 'deit', 'swin', 'cait', 'pvt', 'pit', 'beit', 'convmixer', 'mvit'],
    }

    dataset_styles = {
        'bigearthnet': {
            'linestyle': '-',
            'marker': 'o',
            'color': '#003f5f'  # Blue variation 1
        },
        'rgb_bigearthnet': {
            'linestyle': '-',
            'marker': 'o',
            'color': '#0074D9'  # Blue variation 2
        },
        'deepglobe': {
            'linestyle': '-',
            'marker': 'o',
            'color': '#7FDBFF'  # Blue variation 3
        },
        'imagenet': {
            'linestyle': ':',
            'marker': '^',
            'color': '#8B0000'  # Red variation 1
        },
        'caltech': {
            'linestyle': ':',
            'marker': '^',
            'color': '#FF4136'  # Red variation 2
        },
        'caltech_120': {
            'linestyle': ':',
            'marker': '^',
            'color': '#FFA07A'  # Red variation 3
        }
    }

    model_type_marker = {
        'cnn': 'x',
        'transformer': '^',
    }
    linewidth_metric = 0.5

    def __init__(self, results_data: pd.DataFrame, output_dir: str = None):
        self.results = results_data
        self.subplot_fig = None
        self.subplot_ax = None
        self.num_non_used_subplots = 0
        self.output_dir = output_dir or self.output_dir

    def create_all_plots(
            self,
            transform_names: list[str] | str | None = None,
            dataset_names: list[str] | str | None = None,
            model_names: list[str] | None = None,
            plot_as_subplots: bool = True,
            plot_individual_models: bool = False,
            tight_layout: bool = True,
            save_name: str | None = None,
            metric_type: str = 'macro'
    ):
        self.log.info(f"Creating plots for [{transform_names} | {dataset_names} | {model_names}]")
        dataset_names, transform_names = self._reformat_input_data_as_lists(dataset_names, transform_names)
        results_data = self.results
        num_plots = len(transform_names)
        for idx, transform_name in enumerate(transform_names):
            fig, ax = self._get_fig_ax(plot_as_subplots, idx, num_plots)
            single_transform_results = results_data[results_data['transform'] == transform_name]
            self.create_single_plot(
                ax=ax,
                transform_data=single_transform_results,
                plot_title=transform_name.upper(),
                dataset_names=dataset_names,
                model_names=model_names,
                plot_individual_models=plot_individual_models,
                metric_type=metric_type,
            )
            if tight_layout:
                fig.tight_layout()
            if not plot_as_subplots:
                save_dir = f"{self.output_dir}/{metric_type}/{transform_name}.png"
                fig.savefig(save_dir, dpi=200)
                self.log.info(f"Saved plot {save_dir}")
        if plot_as_subplots:
            self._make_not_used_subplots_invisible(transform_names, num_plots)
            save_name = 'all_transforms' if save_name is None else save_name
            save_dir = f"{self.output_dir}/{save_name}_{metric_type}.png"
            self.subplot_fig.savefig(save_dir, dpi=200)
            self.log.info(f"Saved plot {save_dir}")
            self.subplot_fig = None
            self.subplot_ax = None

    def create_single_plot(
            self,
            ax: plt.Axes,
            transform_data: pd.DataFrame,
            plot_title: str,
            dataset_names: list[str],
            model_names: list[str] | None,
            plot_individual_models: bool = False,
            metric_type: str = 'macro'
    ):
        self.log.info(f"Plotting {plot_title}")
        for dataset_name in dataset_names:
            dataset_results = transform_data[transform_data['dataset'] == dataset_name]
            self.plot_dataset(
                ax=ax,
                dataset_name=dataset_name,
                results=dataset_results,
                model_names=model_names,
                plot_individual_models=plot_individual_models,
                metric_type=metric_type,
            )
        ax.legend()
        ax.set_title(plot_title)

    def plot_dataset(
            self,
            ax: plt.Axes,
            dataset_name: str,
            results: pd.DataFrame,
            plot_individual_models: bool = False,
            model_names: list[str] | None = None,
            metric_type: str = 'macro',
    ) -> None:
        dataset_results = results[results['dataset'] == dataset_name]
        if dataset_results.empty:
            return
        dataset_color = self.dataset_styles[dataset_name]['color']
        if plot_individual_models:
            self.plot_dataset_multiple_models(ax, dataset_name, model_names, dataset_results, metric_type)
        self.plot_dataset_average(ax, dataset_results, dataset_name, dataset_color, metric_type)
        self._set_dataset_plot_layout(ax, dataset_results)

    def plot_dataset_average(
            self,
            ax: plt.Axes,
            dataset_results: pd.DataFrame,
            dataset: str,
            color: str | None = None,
            metric_type: str = 'macro'
    ) -> None:
        dataset_group = dataset_results.groupby('transform_param')
        score_name = 'score'  # _' + metric_type
        dataset_mean = dataset_group.agg({score_name: 'mean'}).reset_index()
        dataset_std = dataset_group.agg({score_name: 'std'}).reset_index()
        marker = self.dataset_styles[dataset]['marker']
        ax.plot(
            dataset_mean['transform_param'],
            dataset_mean[score_name],
            marker=marker,
            linestyle=self.dataset_styles[dataset]['linestyle'],
            label=f"{dataset}",
            color='black' if color is None else color,
            zorder=100
        )
        # plot calculated std around the mean as candlestick
        ax.errorbar(
            dataset_mean['transform_param'],
            dataset_mean[score_name],
            alpha=0.7,
            yerr=dataset_std[score_name],
            fmt=marker,
            color='black' if color is None else color,
            zorder=100,
            capsize=4
        )
        if 'transform_param_labels' in dataset_mean.columns:
            ax.set_xticks(dataset_mean['transform_param'])
            ax.set_xticklabels(dataset_mean['transform_param_labels'])

    def plot_dataset_multiple_models(
            self,
            ax: plt.Axes,
            dataset_name: str,
            model_names: list[str] | None,
            dataset_results: pd.DataFrame,
            metric_type: str = 'macro',
            show_line_style_in_legend: bool = False
    ):
        model_names = model_names or dataset_results['model'].unique()
        dataset_line_style = self.dataset_styles[dataset_name]['linestyle']
        for model in model_names:
            try:
                self.plot_dataset_single_model(
                    ax, model, dataset_results, dataset_line_style, metric_type=metric_type
                )
            except ValueError:
                self.log.error(f"Model {model} not found in dataset {dataset_name}")
        if show_line_style_in_legend:
            ax.plot([], [], linestyle=dataset_line_style, label=dataset_name, color='black')

    def plot_dataset_single_model(
            self,
            ax: plt.Axes,
            model_name: str,
            dataset_results: pd.DataFrame,
            line_styles: str,
            color: str | None = None,
            metric_type: str = 'macro',
    ) -> None:
        score_name = 'score'  # _'+ metric_type
        model_results = dataset_results[dataset_results['model'] == model_name]
        # model_type = esultsExtractor().get_model_type(model)
        x_values = model_results['transform_param']
        y_values = model_results[score_name]
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        ax.plot(
            x_values,
            y_values,
            marker='o',  # self.marker.get(model_type, 'o'),
            linestyle=line_styles,
            label=model_name,
            color=color,
        )

    def _set_dataset_plot_layout(self, ax: plt.Axes, dataset_results: pd.DataFrame) -> None:
        x_ticks = dataset_results['transform_param'].unique()
        x_ticks = sorted(list(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_yticks([i * 0.1 for i in range(0, 11)])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(f"Model Performances (Acc or mAP macro)")
        for y_value in [i * 0.1 for i in range(1, 10)]:
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=self.linewidth_metric)

    def _reformat_input_data_as_lists(
            self,
            dataset_names: list[str] | str | None,
            transform_names: list[str] | str | None,
    ) -> tuple[list[str], list[str]]:
        dataset_names = self.all_dataset_names if dataset_names is None else dataset_names
        dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        transform_names = self.all_transform_names if transform_names is None else transform_names
        transform_names = transform_names if isinstance(transform_names, list) else [
            transform_names]
        return dataset_names, transform_names

    def _get_fig_ax(
            self,
            plot_as_subplots: bool,
            idx: int,
            num_plots: int
    ) -> tuple[plt.Figure, plt.Axes]:
        if plot_as_subplots:
            return self._get_subplot_fig_ax(idx, num_plots)
        else:
            return plt.subplots(1, 1, figsize=(6, 6))

    def _get_subplot_fig_ax(
            self,
            subplot_idx: int,
            total_plots: int
    ) -> tuple[plt.Figure, plt.Axes]:
        nrows = 4 if total_plots > 15 else 3 if total_plots > 6 else 2 if total_plots > 1 else 1
        ncols = (total_plots - 1) // nrows + 1
        total_subplots = nrows * ncols
        self.num_non_used_subplots = total_subplots - total_plots
        if self.subplot_fig is None:
            self.subplot_fig, self.subplot_ax = plt.subplots(nrows, ncols,
                                                             figsize=(5 * ncols, 5 * nrows))
        if ncols > 1:
            ax = self.subplot_ax[subplot_idx // ncols, subplot_idx % ncols]
        elif nrows > 1:
            ax = self.subplot_ax[subplot_idx]
        else:
            ax = self.subplot_ax
        return self.subplot_fig, ax

    def _make_not_used_subplots_invisible(
            self,
            transform_names: list[str],
            num_plots: int
    ) -> None:
        for idx in range(self.num_non_used_subplots):
            fig, ax = self._get_subplot_fig_ax(len(transform_names) + idx, num_plots)
            ax.axis('off')


def prepare_data_for_plotting(
        data_path='C:/CODE/master-thesis/output/results.csv',
        filter_for_transforms: str | None = None,
        filter_out_params: list[int] | None = None,
):
    if filter_out_params is None:
        filter_out_params = [31]
    data = pd.read_csv(data_path)
    data = data.dropna()
    if filter_for_transforms == 'combined':
        data = data[data['transform'].str.contains('~')]
        data['transform_param_labels'] = data['transform_param'].astype(str)
        data['transform_param'] = data['transform_param'].apply(lambda x: float(x.split('~')[0]))
    if filter_for_transforms == 'single':
        data = data[~data['transform'].str.contains('~')]
        data['transform_param_labels'] = data['transform_param'].astype(str)

    for param in filter_out_params:
        data = data[data['transform_param'] != param]

    data['transform_param'] = data['transform_param'].astype(float)
    not_enough_channels_condition = (
            (data['dataset'] != 'bigearthnet') &
            (data['transform'].str.startswith('channel', na=False)) &
            (data['transform_param'] > 3)
    )
    data = data[~not_enough_channels_condition]

    return data


def plot_default_single_transforms_by_type(
        data_path: str = 'C:/CODE/master-thesis/output/transform_results/results.csv',
        output_dir: str = 'C:/CODE/master-thesis/output/results_single'
):
    results = prepare_data_for_plotting(data_path=data_path, filter_for_transforms='single')
    evaluator = ResultsPlotter(results_data=results, output_dir=output_dir)
    for metric_type in ['macro', 'micro']:
        evaluator.create_all_plots(
            save_name='ALL_RESULTS', metric_type=metric_type
        )
        evaluator.create_all_plots(
            save_name='TEXTURE',
            transform_names=['bilateral', 'median', 'gaussian'],
            metric_type=metric_type,
        )
        evaluator.create_all_plots(
            save_name='SHAPE',
            transform_names=['patch_shuffle', 'patch_rotation'],
            metric_type=metric_type,
        )
        evaluator.create_all_plots(
            save_name='COLOR',
            transform_names=['channel_shuffle', 'channel_inversion', 'greyscale'],
            metric_type=metric_type,
        )


def plot_default_combined_transforms(
        data_path: str = 'C:/CODE/master-thesis/output/transform_results/results.csv',
        output_dir: str = 'C:/CODE/master-thesis/output/results_combined',
        transform_combis: list[str] | None = None,
):
    def get_default_combinations():
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

    transform_combis = transform_combis or get_default_combinations()
    results = prepare_data_for_plotting(data_path=data_path, filter_for_transforms='combined')
    evaluator = ResultsPlotter(results_data=results, output_dir=output_dir)
    evaluator.create_all_plots(
        save_name=f'combined/',
        transform_names=transform_combis,
        plot_as_subplots=False,
    )


def plot_single_transform_by_model_type(
        transform_name: str,
        dataset_names: list[str] | None = None,
        data_path: str = 'C:/CODE/master-thesis/output/transform_results/results.csv',
):
    results = prepare_data_for_plotting(data_path=data_path, filter_for_transforms='single')
    evaluator = ResultsPlotter(
        results_data=results, output_dir='C:/CODE/master-thesis/output/results_type_split'
    )
    evaluator.create_all_plots(
        transform_names=transform_name,
        dataset_names=dataset_names,
        model_names=evaluator.models['cnn'],
        plot_individual_models=False,
        save_name=f'{transform_name}_cnn',
    )
    evaluator.create_all_plots(
        transform_names=transform_name,
        dataset_names=dataset_names,
        model_names=evaluator.models['transformer'],
        plot_individual_models=False,
        save_name=f'{transform_name}_transformer',
    )



if __name__ == '__main__':
    # set logger lvl info
    import logging
    logging.basicConfig(level=logging.INFO)
    plot_single_transform_by_model_type('bilateral', 'imagenet')
    exit()
    plot_default_single_transforms_by_type()
    plot_default_combined_transforms()
