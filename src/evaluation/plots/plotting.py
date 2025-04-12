import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from common_utils.logger import create_logger

from evaluation.results.reader import ResultsReader

transform_names_friendly = {
    'channel_shuffle': 'Channel Shuffle',
    'channel_inversion': 'Channel Inversion',
    'greyscale': 'Channel Averaging',
    'bilateral': 'Bilateral',
    'median': 'Median',
    'gaussian': 'Gaussian',
    'patch_shuffle': 'Patch Shuffle',
    'patch_rotation': 'Patch Rotation',
    'noise': 'Noise',
    'bilateral~patch_shuffle': 'Bilateral & Patch Shuffle',
    'bilateral~channel_shuffle': 'Bilateral & Channel Shuffle',
    'patch_shuffle~channel_shuffle': 'Patch Shuffle & Channel Shuffle',
}
transform_names_data = {
    'channel_shuffle': 'Channel Shuffle',
    'channel_inversion': 'Channel Inversion',
    'greyscale': 'Channel Mean',
    'bilateral': 'Bilateral',
    'median': 'Median',
    'gaussian': 'Gaussian',
    'patch_shuffle': 'Patch Shuffle',
    'patch_rotation': 'Patch Rotation',
    'noise': 'Noise',
    'bilateral~patch_shuffle': 'Bilateral~Patch Shuffle',
    'bilateral~channel_shuffle': 'Bilateral~Channel Shuffle',
    'patch_shuffle~channel_shuffle': 'Patch Shuffle~Channel Shuffle',
}
dataset_names_friendly = {
    'imagenet': 'ImageNet',
    'caltech': 'Caltech',
    'caltech_120': 'Caltech 120',
    'caltech_ft': 'Caltech Finet.',
    'bigearthnet': 'BigEarthNet',
    'rgb_bigearthnet': 'RGB BigEarthNet',
    'deepglobe': 'DeepGlobe',
}


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = "C:/CODE/master-thesis/output/test"
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
        'BEN': ['bigearthnet', 'rgb_bigearthnet'],
        'CAL_FT': ['caltech', 'caltech_ft', 'imagenet'],
        'ALL': ['bigearthnet', 'rgb_bigearthnet', 'deepglobe', 'imagenet', 'caltech', 'caltech_120', 'caltech_ft'],
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
            'linestyle': '-',
            'marker': '^',
            'color': '#8B0000'  # Red variation 1
        },
        'caltech': {
            'linestyle': '-',
            'marker': '^',
            'color': '#FF4136'  # Red variation 2
        },
        'caltech_120': {
            'linestyle': '-',
            'marker': '^',
            'color': '#FFA07A'  # Red variation 3
        },
        'caltech_ft': {
            'linestyle': '-',
            'marker': '^',
            'color': '#FF0000'  # Red variation 3
        },
    }
    model_type_styles = {
        'cnn': {
            'linestyle': '-',
            'marker': 'o',
        },
        'transformer': {
            'linestyle': ':',
            'marker': '^',
        }
    }
    _linewidth_metric = 0.5
    ax_label_fontsize = 12
    title_fontsize = 15
    legend_fontsize = 11
    errorbar_default_style = {
        'alpha': 0.5,
        'zorder': 100,
        'capsize': 3,
    }
    x_label = 'Transformation Intensity'
    x_labels = {
        'Noise': 'Noise Intensity',
        'Patch Shuffle': 'Grid Size',
        'Patch Rotation': 'Grid Size',
        'Gaussian': 'Standard Deviation',
        'Bilateral': 'Kernel Diameter',
        'Median': 'Kernel Size',
        'Channel Shuffle': 'Share of Shuffled Channels',
        'Channel Inversion': 'Share of Inverted Channels',
        'Channel Mean': 'Channel Averaging Factor',
        'Bilateral~Patch Shuffle': 'Kernel Diameter & Grid Size',
        'Bilateral~Channel Shuffle': 'Kernel Diameter & Shuffled Channels',
        'Patch Shuffle~Channel Shuffle': 'Grid Size & Shuffled Channels',
    }
    y_labels = {
        'relative_loss': 'Relative Loss of Model Performance',
        'absolute_loss': 'Absolute Loss of Model Performance',
        'score': 'Model Performances (Acc. / mAP macro)',
        'relative_score': 'Relative ',
        'cleaned_score': 'Relative ',
    }


    def __init__(
            self,
            data_path: str | None = None,
            filter_for_transforms: str | None = 'single',
            output_dir: str = None,
            plot_as_subplots: bool = True,
            plot_individual_models: bool = False,
            plot_split_by_model_type: bool = False,
            tight_layout: bool = False,
            metric_type: str = 'macro',
            score_type: str = 'relative_loss',
    ):
        self.data_path = data_path or self.data_path
        self.y_label = self.y_labels[score_type]
        self.metric_type = metric_type
        self.results = ResultsReader().read_data(data_path, filter_for_transforms)
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
        if self.plot_split_by_model_type:
            self.errorbar_default_style['alpha'] = 0.0
        else:
            self.errorbar_default_style['alpha'] = 0.5
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
            transform_name = transform_names_data[transform_name]
            fig, ax = self._get_fig_ax(idx, num_plots)
            single_transform_results = results_data[results_data['transform'] == transform_name]
            self.create_single_plot(ax, single_transform_results, transform_name, dataset_names, model_names)
            if self.tight_layout:
                fig.tight_layout(pad=3.7)
            if not self.plot_as_subplots:
                self._save_plot(fig, ax, save_name=f"{save_name}/{transform_name}", transform_name=transform_name)
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
            self.plot_single_dataset(
                ax=ax,
                dataset_name=dataset_name,
                dataset_results=dataset_results,
                model_names=model_names,
            )
        try:
            plot_title = transform_names_friendly[plot_title]
        except KeyError:
            pass
        ax.set_title(plot_title, fontdict={'fontsize': self.title_fontsize})
        # add a legend entry for the model types styles
        if self.plot_split_by_model_type:
            cnn_style = self.model_type_styles['cnn']
            transformer_style = self.model_type_styles['transformer']
            cnn_style['color'] = 'black'
            transformer_style['color'] = 'black'
            ax.plot([], [], label=' ', color='white')
            ax.plot([], [], **cnn_style, label='CNN')
            ax.plot([], [], **transformer_style, label='Transformer')

    def plot_single_dataset(
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
            self.plot_single_dataset_multiple_models(ax, model_names, dataset_results, dataset_linestyle)

        if self.plot_split_by_model_type:
            self.plot_single_dataset_avg_split_by_architecture(ax, dataset_results, dataset_name)
        else:
            plot_style = self.dataset_styles[dataset_name]
            self.plot_single_dataset_average(
                ax=ax, dataset_results=dataset_results, label=dataset_name, plot_style=plot_style
            )
        self._set_dataset_plot_layout(ax, x_ticks=dataset_results['transform_param'].unique())


    def plot_single_dataset_avg_split_by_architecture(
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
        self.plot_single_dataset_average(
            ax=ax,
            dataset_results=cnn_results,
            label=dataset_name,
            plot_style=cnn_style
        )
        self.plot_single_dataset_average(
            ax=ax,
            dataset_results=transformer_results,
            plot_style=transformer_style
        )

    def plot_single_dataset_average(
            self,
            ax: plt.Axes,
            dataset_results: pd.DataFrame,
            plot_style: dict,
            label: str | None = None,
    ) -> None:
        dataset_grouped = dataset_results.groupby('transform_param')
        dataset_mean = dataset_grouped.agg({self.score_type: 'mean'}).reset_index()
        dataset_std = dataset_grouped.agg({self.score_type: 'std'}).reset_index()
        dataset_names = dataset_results[
            ['transform_param', 'transform_param_labels']].drop_duplicates()
        dataset_mean = dataset_mean.merge(dataset_names, on='transform_param', how='left')
        dataset_std = dataset_std.merge(dataset_names, on='transform_param', how='left')
        dataset_label = dataset_names_friendly[label] if label is not None else None
        ax.plot(
            dataset_mean['transform_param'],
            dataset_mean[self.score_type],
            label=dataset_label,
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
        try:
            if 'transform_param_labels' in dataset_mean.columns:
                if dataset_results['dataset'].iloc[0] not in ['rgb_bigearthnet', 'deepglobe']:
                    # divide the second value after ~ with the number of channels
                    if 'Channel' in dataset_results['transform'].iloc[0]:
                        num_channels = ResultsReader.dataset_channels[dataset_results['dataset'].iloc[0]]
                        dataset_mean['transform_param_labels'] = dataset_mean['transform_param_labels'].apply(
                            lambda x: f"{x.split('~')[0]}~{min(int(x.split('~')[1]), num_channels)}"
                        )

                    # replace ~ with , for better readability
                    dataset_mean['transform_param_labels'] = dataset_mean['transform_param_labels'].apply(
                        lambda x: x.replace('~', ',')
                    )
                    # get value before , for x axis
                    dataset_mean['transform_param_labels_1'] = dataset_mean['transform_param_labels'].apply(
                        lambda x: x.split(',')[0]
                    )
                    dataset_mean['transform_param_labels_2'] = dataset_mean['transform_param_labels'].apply(
                        lambda x: x.split(',')[1] if ',' in x else None
                    )
                    ax.set_xticks(dataset_mean['transform_param'])
                    ax.set_xticklabels(dataset_mean['transform_param_labels'])
        except:
            pass
                #ax.set_xticklabels(dataset_mean['transform_param_labels_1'])
                # test second x axis
                #ax2 = ax.twiny()
                #ax2.set_xticks(dataset_mean['transform_param'])
                #ax.set_xticklabels(dataset_mean['transform_param_labels_2'])

    def plot_single_dataset_multiple_models(
            self,
            ax: plt.Axes,
            model_names: list[str],
            dataset_results: pd.DataFrame,
            line_style: str = '-',
    ):
        for model_name in model_names:
            model_results = dataset_results[dataset_results['model'] == model_name]
            if model_results.empty:
                continue
            x_values = model_results['transform_param']
            y_values = model_results[self.score_type]
            # sort the values by x_values
            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
            ax.plot(
                x_values,
                y_values,
                label=model_name,
                linestyle=line_style,
            )


    def _save_plot(self, fig: plt.Figure, ax: plt.Axes, save_name: str, transform_name: str = None) -> None:
        handles, labels = ax.get_legend_handles_labels()
        loc = 'lower right' if self.num_non_used_subplots != 0 else 'upper right'
        if self.plot_as_subplots is not True:
            loc = 'lower left'
        # fig.legend(handles=handles, labels=labels, loc=loc, fontsize=self.legend_fontsize)
        x_label = self.x_labels[transform_name] if transform_name in self.x_labels else self.x_label
        y_label = f"{self.y_label}{os.getenv('Y_LABEL', '')}Performance"
        fig.supxlabel(x_label, fontsize=self.ax_label_fontsize)
        fig.supylabel(y_label, fontsize=self.ax_label_fontsize)
        if self.plot_as_subplots:
            self.subplot_fig = None
            self.subplot_ax = None
        if '~' in save_name:
            save_name_last = save_name.split('/')[-1]
            present_transform_names = save_name_last.split('~')
            present_transform_categories = [category for category in self.transform_categories.keys() if any(
                transform in present_transform_names for transform in self.transform_categories[category])]
            missing_transform_category = [category for category in self.transform_categories.keys() if category not in present_transform_categories][0]
            save_name_last = f"{missing_transform_category.capitalize()}_{save_name_last}"
            save_name = '/'.join(save_name.split('/')[:-1]) + '/' + save_name_last
        save_dir = f"{self.output_dir}/{save_name}.png"
        fig.savefig(save_dir, dpi=200)

    def _set_dataset_plot_layout(self, ax: plt.Axes, x_ticks: pd.DataFrame) -> None:
        x_ticks = sorted(list(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_yticks([i * 0.1 for i in range(0, 11)])
        ax.set_ylim(bottom=-0.02, top=1.02)
        for y_value in [i * 0.1 for i in range(0, 11)]:
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

    def _get_fig_ax(self, idx: int, num_plots: int) -> tuple[plt.Figure, plt.Axes]:
        if self.plot_as_subplots:
            return self._get_subplot_fig_ax(idx, num_plots)
        else:
            return plt.subplots(1, 1, figsize=(6, 6))

    def _get_subplot_fig_ax(
            self,
            subplot_idx: int,
            total_plots: int
    ) -> tuple[plt.Figure, plt.Axes]:
        num_columns = 4 if total_plots > 15 else 3 if total_plots > 4 else 2 if total_plots > 2 else 1
        num_rows = (total_plots - 1) // num_columns + 1
        total_subplots = num_rows * num_columns
        self.num_non_used_subplots = total_subplots - total_plots
        if self.subplot_fig is None:
            self.subplot_fig, self.subplot_ax = plt.subplots(num_rows, num_columns,
                                                             figsize=(5 * num_columns, 5 * num_rows))
        if num_rows > 1 and num_columns > 1:
            ax = self.subplot_ax[subplot_idx // num_columns, subplot_idx % num_columns]
        elif num_columns > 1 or num_rows > 1:
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
    for score_type in ['relative_score']:
        plotter = ResultsPlotter(
            score_type=score_type,
            data_path='C:/CODE/master-thesis/data/results_v4.csv',
            plot_split_by_model_type=True,
        )
        plotter.create_all_plots(
            transform_names=['noise'],
            # dataset_names=['deepglobe'],
            save_name=f"TEST_NOISE"
        )
