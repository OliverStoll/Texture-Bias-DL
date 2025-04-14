import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from common_utils.logger import create_logger

from evaluation.plots.constants import transform_names_friendly, transform_names_data, dataset_names_friendly
from evaluation.results.reader import ResultsReader


class ResultsPlotter:
    """Handles configuration and setup for plotting model performance under different input transformations."""

    log = create_logger("Transform Plotter")
    data_path = '/data/results_v1.csv'
    _linewidth_metric = 0.5
    ax_label_fontsize = 12
    title_fontsize = 15
    legend_fontsize = 11
    errorbar_default_style = {
        'alpha': 0.5,
        'zorder': 100,
        'capsize': 3,
    }
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
        'cnn': [
            'resnet', 'efficientnet', 'convnext', 'regnet', 'densenet',
            'resnext', 'mobilenet', 'xception', 'inception', 'regnety'
        ],
        'transformer': ['vit', 'deit', 'swin', 'cait', 'pvt', 'pit', 'beit', 'convmixer', 'mvit'],
    }

    dataset_styles = {
        'bigearthnet': {'linestyle': '-', 'marker': 'o', 'color': '#003f5f'},
        'rgb_bigearthnet': {'linestyle': '-', 'marker': 'o', 'color': '#0074D9'},
        'deepglobe': {'linestyle': '-', 'marker': 'o', 'color': '#7FDBFF'},
        'imagenet': {'linestyle': '-', 'marker': '^', 'color': '#8B0000'},
        'caltech': {'linestyle': '-', 'marker': '^', 'color': '#FF4136'},
        'caltech_120': {'linestyle': '-', 'marker': '^', 'color': '#FFA07A'},
        'caltech_ft': {'linestyle': '-', 'marker': '^', 'color': '#FF0000'},
    }
    model_type_styles = {
        'cnn': {'linestyle': '-', 'marker': 'o'},
        'transformer': {'linestyle': ':', 'marker': '^'}
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
        plot_as_subplots: bool = True,
        plot_individual_models: bool = False,
        plot_split_by_model_type: bool = False,
        tight_layout: bool = False,
        metric_type: str = 'macro',
        score_type: str = 'relative_loss',
        output_dir: str = "C:/CODE/master-thesis/output",
    ):
        """
        Initialize a ResultsPlotter instance.

        Args:
            data_path: Optional custom path to the input results CSV file.
            filter_for_transforms: Transformation filter ('single', 'combined', etc.).
            output_dir: Optional directory to save the generated plots.
            plot_as_subplots: Whether to render dataset plots in a subplot grid layout.
            plot_individual_models: Whether to generate separate plots for each model.
            plot_split_by_model_type: Whether to differentiate CNNs and Transformers visually.
            tight_layout: Whether to apply tight layout to matplotlib figures.
            metric_type: Aggregation type for the score metrics (e.g., 'macro').
            score_type: Score key to visualize (e.g., 'relative_loss', 'absolute_loss').
        """
        self.data_path = data_path or self.data_path
        self.y_label = self.y_labels[score_type]
        self.metric_type = metric_type
        self.results = ResultsReader().read_data(data_path, filter_for_transforms)
        self.score_type = score_type
        self.subplot_fig = None
        self.subplot_ax = None
        self.num_non_used_subplots = 0
        self.save_path = output_dir or self.save_path
        self.all_model_names = self.model_categories['cnn'] + self.model_categories['transformer']
        self.all_dataset_names = self.dataset_categories['RS'] + self.dataset_categories['CV']
        self.all_transform_names = list(itertools.chain(*self.transform_categories.values()))
        self.plot_as_subplots = plot_as_subplots
        self.plot_individual_models = plot_individual_models
        self.plot_split_by_model_type = plot_split_by_model_type
        self.tight_layout = tight_layout
        if self.plot_split_by_model_type:
            self.errorbar_default_style['alpha'] = 0.0

        os.makedirs(self.save_path, exist_ok=True)

    def create_all_plots(
            self,
            transform_names: list[str] | str | None = None,
            dataset_names: list[str] | str | None = None,
            model_names: list[str] | str | None = None,
            save_name: str | None = None,
    ):
        """
        Create and optionally save plots for each transformation.

        Args:
            transform_names: Name(s) of the transformations to plot. If None, all are used.
            dataset_names: Name(s) of datasets to include in the plots. If None, all are used.
            model_names: Name(s) of models to include in the plots. If None, all are used.
            save_name: Optional subdirectory name to store individual plots.
        """
        self.log.debug(f"Creating plots for [{transform_names} | {dataset_names} | {model_names}]\n")

        transform_names, dataset_names, model_names = self._reformat_all_inputs_as_lists(
            transform_names, dataset_names, model_names
        )

        results_data = self.results
        num_transforms = len(transform_names)

        for index, original_name in enumerate(transform_names):
            mapped_name = transform_names_data[original_name]
            fig, ax = self._get_fig_ax(index, num_transforms)
            filtered_results = results_data[results_data['transform'] == mapped_name]

            self.create_single_plot(ax, filtered_results, mapped_name, dataset_names, model_names)

            if self.tight_layout:
                fig.tight_layout(pad=3.7)

            if not self.plot_as_subplots:
                self._save_plot(fig, ax, save_name=f"{save_name}/{mapped_name}", transform_name=mapped_name)

        if self.plot_as_subplots:
            self._make_not_used_subplots_invisible(transform_names, num_transforms)
            self._save_plot(fig=self.subplot_fig, ax=ax, save_name=save_name)


    def create_single_plot(
            self,
            ax: plt.Axes,
            transform_data: pd.DataFrame,
            plot_title: str,
            dataset_names: list[str],
            model_names: list[str],
    ):
        """
        Create a single plot for a given transformation and set of datasets.

        Args:
            ax: Matplotlib Axes object to draw the plot on.
            transform_data: DataFrame containing results for the current transformation.
            plot_title: Transformation name (used as plot title).
            dataset_names: List of dataset names to include in the plot.
            model_names: List of model names to filter results.
        """
        filtered_data = transform_data[transform_data['model'].isin(model_names)]

        for dataset_name in dataset_names:
            dataset_subset = filtered_data[filtered_data['dataset'] == dataset_name]
            self.plot_single_dataset(
                ax=ax,
                dataset_name=dataset_name,
                dataset_results=dataset_subset,
                model_names=model_names,
            )

        ax.set_title(plot_title, fontdict={'fontsize': self.title_fontsize})

        if self.plot_split_by_model_type:
            cnn_style = self.model_type_styles['cnn'].copy()
            transformer_style = self.model_type_styles['transformer'].copy()
            cnn_style['color'] = 'black'
            transformer_style['color'] = 'black'

            ax.plot([], [], label=' ', color='white')  # empty spacer for alignment
            ax.plot([], [], **cnn_style, label='CNN')
            ax.plot([], [], **transformer_style, label='Transformer')

    def plot_single_dataset(
            self,
            ax: plt.Axes,
            dataset_name: str,
            dataset_results: pd.DataFrame,
            model_names: list[str],
    ) -> None:
        """
        Plot performance for a single dataset under a given transformation.

        Args:
            ax: Matplotlib Axes object to draw the plot on.
            dataset_name: Name of the dataset being plotted.
            dataset_results: DataFrame containing filtered results for the dataset.
            model_names: List of model names to include in the plot.
        """
        if dataset_results.empty:
            return

        dataset_linestyle = self.dataset_styles[dataset_name]['linestyle']

        if self.plot_individual_models:
            self.plot_single_dataset_multiple_models(
                ax=ax,
                model_names=model_names,
                dataset_results=dataset_results,
                line_style=dataset_linestyle
            )

        if self.plot_split_by_model_type:
            self.plot_single_dataset_avg_split_by_architecture(
                ax=ax,
                dataset_results=dataset_results,
                dataset_name=dataset_name
            )
        else:
            plot_style = self.dataset_styles[dataset_name]
            self.plot_single_dataset_average(
                ax=ax,
                dataset_results=dataset_results,
                label=dataset_name,
                plot_style=plot_style
            )

        self._set_dataset_plot_layout(ax, x_ticks=dataset_results['transform_param'].unique())


    def plot_single_dataset_avg_split_by_architecture(
            self,
            ax: plt.Axes,
            dataset_results: pd.DataFrame,
            dataset_name: str
    ) -> None:
        """
        Plot average performance of CNN and Transformer models separately for a given dataset.

        Args:
            ax: Matplotlib Axes object to draw the plot on.
            dataset_results: DataFrame containing model results for the dataset.
            dataset_name: Name of the dataset being plotted.
        """
        cnn_results = dataset_results[dataset_results['model'].isin(self.model_categories['cnn'])]
        transformer_results = dataset_results[dataset_results['model'].isin(self.model_categories['transformer'])]

        base_color = self.dataset_styles[dataset_name]['color']
        cnn_style = self.model_type_styles['cnn'].copy()
        transformer_style = self.model_type_styles['transformer'].copy()

        cnn_style['color'] = base_color
        transformer_style['color'] = base_color

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
        """
        Plot the average performance with standard deviation error bars for a single dataset.

        Args:
            ax: Matplotlib Axes object to draw the plot on.
            dataset_results: DataFrame containing model results for the dataset.
            plot_style: Dictionary defining line and marker style for the plot.
            label: Optional label for the dataset (used in legend).
        """
        dataset_grouped = dataset_results.groupby('transform_param')
        dataset_mean = dataset_grouped.agg({self.score_type: 'mean'}).reset_index()
        dataset_std = dataset_grouped.agg({self.score_type: 'std'}).reset_index()
        dataset_names = dataset_results[['transform_param', 'transform_param_labels']].drop_duplicates()
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
        self._check_to_split_transform_param_labels(ax, dataset_mean, dataset_results)

    def _check_to_split_transform_param_labels(
            self,
            ax: plt.Axes,
            dataset_mean: pd.DataFrame,
            dataset_results: pd.DataFrame
    ) -> None:
        """
        Conditionally split and apply readable transformation parameter labels to the x-axis.

        Args:
            ax: Matplotlib Axes object to apply x-tick labels to.
            dataset_mean: DataFrame with aggregated means and transform label columns.
            dataset_results: Full results DataFrame to inspect dataset and transform context.
        """
        if (
            'transform_param_labels' in dataset_mean.columns and
            '~' in dataset_mean['transform_param_labels'].iloc[0] and
            dataset_results['dataset'].iloc[0] not in ['rgb_bigearthnet', 'deepglobe']
        ):
            try:
                self._split_transform_param_labels(ax, dataset_mean, dataset_results)
            except (KeyError, IndexError, ValueError) as e:
                self.log.warning(f"Failed to split labels for dataset='{dataset_results['dataset'].iloc[0]}': {e}")

    @staticmethod
    def _split_transform_param_labels(ax: plt.Axes, dataset: pd.DataFrame, dataset_results: pd.DataFrame) -> None:
        """
        Split and apply readable x-axis labels from paired transformation parameter labels.

        Args:
            ax: Matplotlib Axes object to apply the labels to.
            dataset: DataFrame containing aggregated means and label columns.
            dataset_results: Original results DataFrame used to extract metadata (e.g., transform, dataset).
        """
        if 'Channel' in dataset_results['transform'].iloc[0]:
            num_channels = ResultsReader.dataset_channels[dataset_results['dataset'].iloc[0]]
            dataset['transform_param_labels'] = dataset['transform_param_labels'].apply(
                lambda x: f"{x.split('~')[0]}~{min(int(x.split('~')[1]), num_channels)}"
            )
        dataset['transform_param_labels'] = dataset['transform_param_labels'].apply(lambda x: x.replace('~', ','))
        dataset['transform_param_labels_1'] = dataset['transform_param_labels'].apply(lambda x: x.split(',')[0])
        dataset['transform_param_labels_2'] = dataset['transform_param_labels'].apply(
            lambda x: x.split(',')[1] if ',' in x else None
        )
        ax.set_xticks(dataset['transform_param'])
        ax.set_xticklabels(dataset['transform_param_labels'])

    def plot_single_dataset_multiple_models(
            self,
            ax: plt.Axes,
            model_names: list[str],
            dataset_results: pd.DataFrame,
            line_style: str = '-',
    ):
        """
        Plot individual model performance reduction curves for a given dataset and transformation.

        Args:
            ax: Matplotlib Axes object to draw the plots on.
            model_names: List of model identifiers to plot.
            dataset_results: DataFrame containing model results filtered for the dataset.
            line_style: Line style to apply to all model plots (default is solid).
        """
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
        """
        Save the generated plot to disk, optionally adjusting naming for dual transformations.

        Args:
            fig: Matplotlib Figure object to save.
            ax: Matplotlib Axes object containing the plot.
            save_name: Base filename (without extension) and optional subdirectory path.
            transform_name: Optional transformation name used to set x-axis label.
        """
        handles, labels = ax.get_legend_handles_labels()
        loc = 'lower right' if self.num_non_used_subplots != 0 else 'upper right'
        if self.plot_as_subplots is not True:
            loc = 'lower left'
        fig.legend(handles=handles, labels=labels, loc=loc, fontsize=self.legend_fontsize)
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
            present_transform_categories = [
                category for category in self.transform_categories.keys()
                if any(
                    transform in present_transform_names for transform in self.transform_categories[category]
                )
            ]
            missing_transform_category = [
                category for category in self.transform_categories.keys()
                if category not in present_transform_categories
            ][0]
            save_name_last = f"{missing_transform_category.capitalize()}_{save_name_last}"
            save_name = '/'.join(save_name.split('/')[:-1]) + '/' + save_name_last
        save_dir = f"{self.save_path}/{save_name}.png"
        fig.savefig(save_dir, dpi=200)

    def _get_subplot_fig_ax(
            self,
            subplot_idx: int,
            total_plots: int
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Generate or retrieve a subplot figure and corresponding axes for the current index.

        Args:
            subplot_idx: Index of the current subplot to retrieve.
            total_plots: Total number of plots that will be rendered.

        Returns:
            A tuple containing the Matplotlib Figure and the corresponding Axes object.
        """
        # Determine subplot grid dimensions
        if total_plots > 15:
            num_columns = 4
        elif total_plots > 4:
            num_columns = 3
        elif total_plots > 2:
            num_columns = 2
        else:
            num_columns = 1

        num_rows = (total_plots - 1) // num_columns + 1
        total_subplots = num_rows * num_columns
        self.num_non_used_subplots = total_subplots - total_plots

        # Create figure and axes only once
        if self.subplot_fig is None:
            self.subplot_fig, self.subplot_ax = plt.subplots(
                num_rows,
                num_columns,
                figsize=(5 * num_columns, 5 * num_rows)
            )

        # Select the correct Axes object
        if num_rows > 1 and num_columns > 1:
            ax = self.subplot_ax[subplot_idx // num_columns, subplot_idx % num_columns]
        elif num_rows > 1 or num_columns > 1:
            ax = self.subplot_ax[subplot_idx]
        else:
            ax = self.subplot_ax

        return self.subplot_fig, ax

    def _get_fig_ax(self, idx: int, num_plots: int) -> tuple[plt.Figure, plt.Axes]:
        """
        Get a figure and axes object for the given subplot index.

        Args:
            idx: Index of the current subplot.
            num_plots: Total number of plots to be created.

        Returns:
            A tuple containing the Matplotlib Figure and Axes objects.
        """
        if self.plot_as_subplots:
            return self._get_subplot_fig_ax(idx, num_plots)
        else:
            return plt.subplots(1, 1, figsize=(6, 6))

    def _make_not_used_subplots_invisible(
            self,
            transform_names: list[str],
            num_plots: int
    ) -> None:
        """
        Disable axis visibility for any unused subplots in a subplot grid.

        Args:
            transform_names: List of all transformation names being plotted.
            num_plots: Total number of plots used (the rest will be hidden).
        """
        for idx in range(self.num_non_used_subplots):
            fig, ax = self._get_subplot_fig_ax(len(transform_names) + idx, num_plots)
            ax.axis('off')

    def _set_dataset_plot_layout(self, ax: plt.Axes, x_ticks: pd.DataFrame) -> None:
        """
        Set consistent layout parameters for the dataset-specific subplot.

        Args:
            ax: Matplotlib Axes object to apply layout settings to.
            x_ticks: Iterable of x-axis tick positions (e.g., transformation intensities).
        """
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
        """
        Ensure that all input arguments are formatted as lists.

        Args:
            transform_names: A single transformation name, a list of names, or None.
            dataset_names: A single dataset name, a list of names, or None.
            model_names: A single model name, a list of names, or None.

        Returns:
            A tuple of three lists: (transform_names, dataset_names, model_names),
            using defaults if any input is None.
        """
        transform_names = self._reformat_input_as_list(transform_names, self.all_transform_names)
        dataset_names = self._reformat_input_as_list(dataset_names, self.all_dataset_names)
        model_names = self._reformat_input_as_list(model_names, self.all_model_names)
        return transform_names, dataset_names, model_names


    @staticmethod
    def _reformat_input_as_list(value: list[str] | str | None, default: list[str]):
        """
        Ensure the input is returned as a list of strings.

        Args:
            value: A string, list of strings, or None.
            default: Default list to use if value is None.

        Returns:
            A list of strings based on the input or default.
        """
        value = default if value is None else value
        value = value if isinstance(value, list) else [value]
        return value



if __name__ == '__main__':
    """ Example Usage """
    data_path = 'C:/CODE/master-thesis/data/run_results.csv'

    for score_type in ['relative_score']:

        plotter = ResultsPlotter(
            score_type=score_type,
            data_path='C:/CODE/master-thesis/data/results_v4.csv',
            plot_split_by_model_type=True,
        )
        plotter.create_all_plots()
