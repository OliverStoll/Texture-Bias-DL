import itertools
import os
from threading import Thread
import pandas as pd
import matplotlib.pyplot as plt
from common_utils.config import CONFIG
from common_utils.logger import create_logger


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = f"{CONFIG['output_dir']}/test"
    results_df_path = f"{output_dir}/results.csv"
    data_path = '/data/results_v1.csv'
    transform_categories = {
        'color': ['channel_shuffle', 'channel_inversion', 'greyscale'],
        'texture': ['bilateral', 'median', 'gaussian'],
        'shape': ['patch_shuffle', 'patch_rotation'],
    }
    dataset_categories = {
        'RS': ['bigearthnet', 'rgb_bigearthnet', 'deepglobe'],
        'CV': ['imagenet', 'caltech', 'caltech_120', 'caltech_ft'],
    }
    model_categories = {
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
            'linestyle': '--',
            'marker': '^',
            'color': '#8B0000'  # Red variation 1
        },
        'caltech': {
            'linestyle': '--',
            'marker': '^',
            'color': '#FF4136'  # Red variation 2
        },
        'caltech_120': {
            'linestyle': '--',
            'marker': '^',
            'color': '#FFA07A'  # Red variation 3
        },
        'caltech_ft': {
            'linestyle': '--',
            'marker': '^',
            'color': '#FF0000'  # Red variation 3
        },
    }
    model_type_styles = {
        'cnn': {
            'linestyle': '-',
            'marker': 'o',
            # 'color': '#2ECC40'  # Green variation 1
        },
        'transformer': {
            'linestyle': ':',
            'marker': '^',
            # 'color': '#FF4136'  # Red variation 2
        }
    }
    _linewidth_metric = 0.5
    errorbar_default_style = {
        'alpha': 0.6,
        'zorder': 100,
        'capsize': 4,
    }
    x_label = 'Intensity'
    y_labels = {
        'relative_loss': 'Relative Loss of Model Performance',
        'absolute_loss': 'Absolute Loss of Model Performance',
        'score': 'Model Performances (Acc or mAP macro)',
        'relative_score': 'Relative Model Performance, compared to no transformation',
    }

    def _filter_experiment(self, data: pd.DataFrame, filter_experiment: str) -> pd.DataFrame:
        if filter_experiment == 'combined':
            data = data[data['transform'].str.contains('~')]
            data['transform_param_labels'] = data['transform_param'].astype(str)
            data['transform_param'] = data['transform_param'].apply(
                lambda x: float(x.split('~')[0]))
        if filter_experiment == 'single':
            data = data[~data['transform'].str.contains('~')]
            data['transform_param_labels'] = data['transform_param'].astype(str)
        return data

    def _clean_unwanted_params(
            self, data: pd.DataFrame, filter_out_params: list[int] | None = None
    ) -> pd.DataFrame:
        data['transform_param'] = data['transform_param'].astype(float)
        for param in filter_out_params or [31]:
            data = data[data['transform_param'] != param]
        return data

    def _clean_not_enough_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        not_enough_channels_condition = (
                (data['dataset'] != 'bigearthnet') &
                (data['transform'].str.startswith('channel', na=False)) &
                (data['transform_param'] > 3)
        )
        return data[~not_enough_channels_condition]

    def _filter_out_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        unique_idx = data.groupby(['dataset', 'model', 'transform', 'transform_param'])['timestamp'].idxmax()
        unique_data = data.loc[unique_idx]
        return unique_data


    def _calculate_other_score_types(self, results: pd.DataFrame) -> pd.DataFrame:
        results['score'] = results['score_' + self.metric_type]
        results = results.sort_values(by=['transform', 'dataset', 'model', 'transform_param'])
        results = results.reset_index(drop=True)
        grouped_scores = results.groupby(['transform', 'dataset', 'model'])['score']
        results['absolute_loss'] = grouped_scores.transform(lambda x: x.iloc[0] - x)
        results['relative_loss'] = grouped_scores.transform(
            lambda x: (x.iloc[0] - x) / x.iloc[0]
        )
        results['relative_score'] = grouped_scores.transform(
            lambda x: x / x.iloc[0]
        )
        return results

    def _prepare_data_for_plotting(
            self,
            data_path: str,
            filter_for_transforms: str = 'single',
            filter_out_params: list[int] | None = None,
    ):
        data = pd.read_csv(data_path)
        data = data.dropna()
        data = self._filter_experiment(data, filter_for_transforms)
        data = self._clean_unwanted_params(data, filter_out_params)
        data = self._clean_not_enough_channels(data)
        data = self._filter_out_duplicates(data)
        data = self._calculate_other_score_types(data)

        return data

    def __init__(
            self,
            data_path: str | None = None,
            filter_for_transforms: str | None = 'single',
            filter_out_params: list[int] | None = None,
            output_dir: str = None,
            plot_as_subplots: bool = True,
            plot_individual_models: bool = False,
            plot_split_by_model_type: bool = False,
            tight_layout: bool = True,
            metric_type: str = 'macro',
            score_type: str = 'relative_loss',
    ):
        self.data_path = data_path or self.data_path
        self.y_label = self.y_labels[score_type]
        self.metric_type = metric_type
        self.results = self._prepare_data_for_plotting(
            data_path, filter_for_transforms, filter_out_params
        )
        self.score_type = score_type
        self.subplot_fig = None
        self.subplot_ax = None
        self.num_non_used_subplots = 0
        self.output_dir = output_dir or self.output_dir
        self.all_model_names = self.model_categories['cnn'] + self.model_categories['transformer']
        self.all_dataset_names = self.dataset_categories['RS'] + self.dataset_categories['CV']
        self.all_transform_names = list(itertools.chain(*self.transform_categories.values()))
        self.plot_as_subplots = plot_as_subplots
        self.plot_individual_models = plot_individual_models
        self.plot_split_by_model_type = plot_split_by_model_type
        self.tight_layout = tight_layout
        os.makedirs(self.output_dir, exist_ok=True)

    def create_all_plots(
            self,
            transform_names: list[str] | str | None = None,
            dataset_names: list[str] | str | None = None,
            model_names: list[str] | str | None = None,
            save_name: str | None = None,
    ):
        self.log.debug(f"Creating plots for [{transform_names} | {dataset_names} | {model_names}]\n")
        transform_names, dataset_names, model_names = self._reformat_all_inputs_as_lists(
            transform_names, dataset_names, model_names
        )
        results_data = self.results
        num_plots = len(transform_names)
        for idx, transform_name in enumerate(transform_names):
            fig, ax = self._get_fig_ax(idx, num_plots)
            single_transform_results = results_data[results_data['transform'] == transform_name]
            self.create_single_plot(ax, single_transform_results, transform_name, dataset_names, model_names)
            if self.tight_layout:
                fig.tight_layout(pad=3.7)
            if not self.plot_as_subplots:
                self._save_plot(fig, ax, save_name=transform_name)
        if self.plot_as_subplots:
            self._make_not_used_subplots_invisible(transform_names, num_plots)
            self._save_plot(fig=self.subplot_fig, ax=ax, save_name=save_name)

    def create_single_plot(
            self,
            ax: plt.Axes,
            transform_data: pd.DataFrame,
            plot_title: str,
            dataset_names: list[str],
            model_names: list[str],
    ):
        transform_data = transform_data[transform_data['model'].isin(model_names)]
        for dataset_name in dataset_names:
            dataset_results = transform_data[transform_data['dataset'] == dataset_name]
            self.plot_dataset(
                ax=ax,
                dataset_name=dataset_name,
                dataset_results=dataset_results,
                model_names=model_names,
            )
        ax.set_title(plot_title)
        # add a legend entry for the model types styles
        if self.plot_split_by_model_type:
            cnn_style = self.model_type_styles['cnn']
            transformer_style = self.model_type_styles['transformer']
            cnn_style['color'] = transformer_style['color'] = 'black'
            ax.plot([], [], **cnn_style, label='CNN')
            ax.plot([], [], **transformer_style, label='Transformer')

    def plot_dataset(
            self,
            ax: plt.Axes,
            dataset_name: str,
            dataset_results: pd.DataFrame,
            model_names: list[str],
    ) -> None:
        dataset_linestyle = self.dataset_styles[dataset_name]['linestyle']
        if dataset_results.empty:
            return
        if self.plot_individual_models:
            self.plot_dataset_multiple_models(ax, model_names, dataset_results, dataset_linestyle)

        if self.plot_split_by_model_type:
            self.plot_dataset_averages_split_by_model_type(ax, dataset_results, dataset_name)
        else:
            plot_style = self.dataset_styles[dataset_name]
            self.plot_dataset_average(
                ax=ax, dataset_results=dataset_results, label=dataset_name, plot_style=plot_style)
        self._set_dataset_plot_layout(ax, x_ticks=dataset_results['transform_param'].unique())

    def plot_model_type_average(self):
        raise NotImplementedError
        # TODO: Implement


    def plot_dataset_averages_split_by_model_type(
            self,
            ax: plt.Axes,
            results: pd.DataFrame,
            dataset_name: str
    ) -> None:
        cnn_results = results[results['model'].isin(self.model_categories['cnn'])]
        transformer_results = results[results['model'].isin(self.model_categories['transformer'])]
        dataset_color = self.dataset_styles[dataset_name]['color']
        transformer_style = self.model_type_styles['transformer']
        cnn_style = self.model_type_styles['cnn']
        cnn_style['color'] = dataset_color
        transformer_style['color'] = dataset_color
        self.plot_dataset_average(
            ax=ax,
            dataset_results=cnn_results,
            label=dataset_name,
            plot_style=cnn_style
        )
        self.plot_dataset_average(
            ax=ax,
            dataset_results=transformer_results,
            plot_style=transformer_style
        )

    def plot_dataset_average(
            self,
            ax: plt.Axes,
            dataset_results: pd.DataFrame,
            plot_style: dict,
            label: str | None = None,
    ) -> None:
        dataset_grouped = dataset_results.groupby('transform_param')
        dataset_mean = dataset_grouped.agg({self.score_type: 'mean'}).reset_index()
        dataset_std = dataset_grouped.agg({self.score_type: 'std'}).reset_index()
        ax.plot(
            dataset_mean['transform_param'],
            dataset_mean[self.score_type],
            label=label,
            **plot_style,
            zorder=100
        )
        ax.errorbar(
            dataset_mean['transform_param'],
            dataset_mean[self.score_type],
            yerr=dataset_std[self.score_type],
            fmt=plot_style['marker'],
            color=plot_style['color'] if 'color' in plot_style else None,
            **self.errorbar_default_style,
        )
        if 'transform_param_labels' in dataset_mean.columns:
            ax.set_xticks(dataset_mean['transform_param'])
            ax.set_xticklabels(dataset_mean['transform_param_labels'])

    def plot_dataset_multiple_models(
            self,
            ax: plt.Axes,
            model_names: list[str],
            dataset_results: pd.DataFrame,
            line_style: str = '-',
    ):
        for model_name in model_names:
            model_results = dataset_results[dataset_results['model'] == model_name]
            if not model_results.empty:
                self.plot_dataset_single_model(ax, model_name, model_results, line_style)

    def plot_dataset_single_model(
            self,
            ax: plt.Axes,
            model_name: str,
            model_results: pd.DataFrame,
            line_style: str = '-',
            marker: str = 'o',
            color: str | None = None,
    ) -> None:
        x_values = model_results['transform_param']
        y_values = model_results[self.score_type]
        # sort the values by x_values
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        ax.plot(
            x_values,
            y_values,
            marker=marker,
            label=model_name,
            linestyle=line_style,
            color=color,
        )

    def _save_plot(self, fig: plt.Figure, ax: plt.Axes, save_name: str):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc='lower right')
        fig.supylabel(self.y_label)
        fig.supxlabel(self.x_label)
        if self.plot_as_subplots:
            self.subplot_fig = None
            self.subplot_ax = None

        # TODO: insert missing feature type if ~ is in the transform name
        if '~' in save_name:
            present_transform_names = save_name.split('/')[-1].split('~')
            present_transform_categories = [category for category in self.transform_categories.keys() if any(
                transform in present_transform_names for transform in self.transform_categories[category])]
            missing_transform_category = [category for category in self.transform_categories.keys() if category not in present_transform_categories][0]
            save_name = f"{missing_transform_category.capitalize()}_{save_name}"
        save_dir = f"{self.output_dir}/{save_name}.png"
        fig.savefig(save_dir, dpi=200)

    def _set_dataset_plot_layout(self, ax: plt.Axes, x_ticks: pd.DataFrame) -> None:
        x_ticks = sorted(list(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_yticks([i * 0.1 for i in range(0, 11)])
        ax.set_ylim(bottom=0)
        # ax.set_xlabel(self.x_label)
        # ax.set_ylabel(self.y_label)
        for y_value in [i * 0.1 for i in range(1, 10)]:
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=self._linewidth_metric)

    def _reformat_all_inputs_as_lists(
            self,
            transform_names: list[str] | str | None,
            dataset_names: list[str] | str | None,
            model_names: list[str] | str | None,
    ) -> tuple[list[str], list[str], list[str]]:
        transform_names = self._reformat_input_as_list(transform_names, self.all_transform_names)
        dataset_names = self._reformat_input_as_list(dataset_names, self.all_dataset_names)
        model_names = self._reformat_input_as_list(model_names, self.all_model_names)
        return transform_names, dataset_names, model_names

    def _reformat_input_as_list(self, input: list[str] | str | None, default: list[str]):
        input = default if input is None else input
        input = input if isinstance(input, list) else [input]
        return input

    def _get_fig_ax(
            self,
            idx: int,
            num_plots: int
    ) -> tuple[plt.Figure, plt.Axes]:
        if self.plot_as_subplots:
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


if __name__ == '__main__':
    for score_type in ['relative_loss', 'absolute_loss', 'score']:
        plotter = ResultsPlotter(
            score_type=score_type,
            data_path='C:/CODE/master-thesis/data/results_v2.csv',
        )
        plotter.create_all_plots(
            transform_names=['channel_shuffle'],
            dataset_names=['deepglobe'],
            save_name=f"TEST_{score_type}_deepglobe"
        )
