import pandas as pd
import matplotlib.pyplot as plt
from common_utils.config import CONFIG
from common_utils.logger import create_logger

# commented out for weird circular import error on windows
# from transforms.transforms_factoryimport TransformFactory
# from datasets import DataLoaderFactory

# from evaluation.results_pipeline import ResultsExtractor


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = f"{CONFIG['output_dir']}/transform_results"
    results_df_path = f"{output_dir}/results.csv"
    # list(TransformFactory().transforms.keys())
    all_transform_names = ['bilateral', 'median', 'gaussian', 'sobel', 'patch_shuffle', 'patch_rotation', 'channel_shuffle', 'channel_inversion', 'greyscale']
    all_dataset_names = ['bigearthnet', 'rgb_bigearthnet', 'deepglobe', 'imagenet', 'caltech', 'caltech_120']
    # list(DataLoaderFactory().dataset_names)
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

    marker = {
        'transformer': '^',
        'cnn': 'x'
    }
    linewidth_metric = 0.5

    def __init__(self, results):
        self.results = results
        self.subplot_fig = None
        self.subplot_ax = None
        self.num_non_used_subplots = 0

    def _clean_dataset_transform_input_data(self, dataset_names, transform_names):
        dataset_names = self.all_dataset_names if dataset_names is None else dataset_names
        dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        transform_names = self.all_transform_names if transform_names is None else transform_names
        transform_names = transform_names if isinstance(transform_names, list) else [transform_names]
        return dataset_names, transform_names

    def _get_subplot_fig_ax(self, idx, num_plots):
        nrows = 3 if num_plots > 6 else 2 if num_plots > 1 else 1
        ncols = (num_plots-1) // nrows + 1
        total_subplots = nrows * ncols
        self.num_non_used_subplots = total_subplots - num_plots
        if self.subplot_fig is None:
            self.subplot_fig, self.subplot_ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        ax = self.subplot_ax[idx % nrows, idx // nrows] if ncols > 1 else self.subplot_ax[idx]
        return self.subplot_fig, ax

    def _get_fig_ax_depending_on_subplots(self, subplots, idx, num_plots):
        if subplots:
            return self._get_subplot_fig_ax(idx, num_plots)
        else:
            return plt.subplots(1, 1, figsize=(10, 6))


    def plot_single_models_results(self, ax, model, dataset_results, line_styles, color=None):
        model_results = dataset_results[dataset_results['model'] == model]
        # model_type = esultsExtractor().get_model_type(model)
        x_values = model_results['transform_param']
        y_values = model_results['score']
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        ax.plot(
            x_values,
            y_values,
            marker='o',  # self.marker.get(model_type, 'o'),
            linestyle=line_styles,
            label=model,
            color=color,
        )

    def plot_dataset_avg_results(self, ax, dataset_results, dataset, color=None):
        dataset_group = dataset_results.groupby('transform_param')
        dataset_mean = dataset_group.agg({'score': 'mean'}).reset_index()
        dataset_std = dataset_group.agg({'score': 'std'}).reset_index()
        self.log.debug(f"\t{dataset:<16} "
                       f"mean: {dataset_mean['score'].to_list()[-1]:.2f} "
                       f"std: {dataset_std['score'].to_list()[-1]:.2f}")
        marker = self.dataset_styles[dataset]['marker']
        ax.plot(
            dataset_mean['transform_param'],
            dataset_mean['score'],
            marker=marker,
            linestyle=self.dataset_styles[dataset]['linestyle'],
            label=f"{dataset}",
            color='black' if color is None else color,
            zorder=100
        )
        # plot calculated std around the mean as candlestick
        ax.errorbar(
            dataset_mean['transform_param'],
            dataset_mean['score'],
            alpha=0.7,
            yerr=dataset_std['score'],
            fmt=marker,
            color='black' if color is None else color,
            zorder=100,
            capsize=4
        )


    def plot_single_datasets_results(self, ax, dataset, results, plot_individual_models, models=None):
        dataset_results = results[results['dataset'] == dataset]
        line_styles = self.dataset_styles[dataset]['linestyle']
        color = self.dataset_styles[dataset]['color']
        models = models or dataset_results['model'].unique()
        if plot_individual_models:
            for model in models:
                try:
                    self.plot_single_models_results(ax, model, dataset_results, line_styles)
                except:
                    self.log.error(f"Model {model} not found in dataset {dataset}")
            ax.plot([], [], linestyle=line_styles, label=dataset, color='black')
        self.plot_dataset_avg_results(ax, dataset_results, dataset, color=color)
        x_ticks = dataset_results['transform_param'].unique()
        x_ticks = sorted(list(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_yticks([i * 0.1 for i in range(0, 11)])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(f"Model Performances (Acc or mAP macro)")
        for y_value in [i * 0.1 for i in range(1, 10)]:
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=self.linewidth_metric)

    def plot_single_plot(self, fig, ax, transform_data, title, datasets, models, plot_individual_models, tight_layout):
        self.log.info(f"Plotting {title}")
        for dataset in datasets:
            dataset_results = transform_data[transform_data['dataset'] == dataset]
            self.plot_single_datasets_results(
                ax=ax,
                dataset=dataset,
                results=dataset_results,
                models=models,
                plot_individual_models=plot_individual_models
            )
        ax.set_title(title)
        ax.legend()
        if tight_layout:
            fig.tight_layout()

    def _make_not_used_subplots_invisible(self, transform_names, num_plots):
        for idx in range(self.num_non_used_subplots):
            fig, ax = self._get_subplot_fig_ax(len(transform_names) + idx, num_plots)
            ax.axis('off')

    def plot_all_plots(
        self,
        transform_names: list[str] | str | None = None,
        dataset_names: list[str] | str | None = None,
        model_names: list[str] | None = None,
        subplots: bool = True,
        plot_individual_models: bool = False,
        tight_layout: bool = True,
        save_name: str | None = None,
    ):
        dataset_names, transform_names = self._clean_dataset_transform_input_data(dataset_names, transform_names)
        results_df = self.results
        num_plots = len(transform_names)
        for idx, transform_name in enumerate(transform_names):
            fig, ax = self._get_fig_ax_depending_on_subplots(subplots, idx, num_plots)
            single_transform_results = results_df[results_df['transform'] == transform_name]
            self.plot_single_plot(
                fig=fig,
                ax=ax,
                transform_data=single_transform_results,
                title=transform_name.upper(),
                datasets=dataset_names,
                models=model_names,
                plot_individual_models=plot_individual_models,
                tight_layout=tight_layout
            )
            if not subplots:
                save_dir = f"{self.output_dir}/{transform_name}.png"
                fig.savefig(save_dir, dpi=200)
                self.log.info(f"Saved plot {save_dir}")
        if subplots:
            self._make_not_used_subplots_invisible(transform_names, num_plots)
            save_name = 'all_transforms' if save_name is None else save_name
            save_dir = f"{self.output_dir}/{save_name}.png"
            self.subplot_fig.savefig(save_dir, dpi=200)
            self.log.info(f"Saved plot {save_dir}")
            self.subplot_fig = None
            self.subplot_ax = None


def plot_default_single_transforms_by_type(results_path='C:/CODE/master-thesis/output/transform_results/results.csv'):
    results = pd.read_csv(results_path)
    # filter results with transform param == 31
    results = results[results['transform_param'] != 31]
    # filter results that have ~ in transform name
    results = results[~results['transform'].str.contains('~')]
    evaluator = ResultsPlotter(results=results)
    evaluator.plot_all_plots(
        save_name='ALL_RESULTS'
    )
    evaluator.plot_all_plots(
        save_name='TEXTURE', transform_names=['bilateral', 'median', 'gaussian']
    )
    evaluator.plot_all_plots(
        save_name='SHAPE', transform_names=['patch_shuffle', 'patch_rotation']
    )
    evaluator.plot_all_plots(
        save_name='COLOR', transform_names=['channel_shuffle', 'channel_inversion', 'greyscale']
    )


if __name__ == '__main__':
    plot_default_single_transforms_by_type(
        results_path='C:/CODE/master-thesis/output/results.csv'
    )
