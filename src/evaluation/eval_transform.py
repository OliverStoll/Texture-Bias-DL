import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from utils.config import ROOT_DIR, CONFIG
from utils.logger import create_logger

from models import ModelFactory
from transforms import TransformFactory


class ResultsExtractor:
    log = create_logger("Transform Evaluation")
    all_runs_dir = f'{ROOT_DIR}logs/wandb'
    output_dir = f"{CONFIG['output_dir']}/transform_results"
    results_df_path = f"{output_dir}/results.csv"
    internal_log_path = 'files/wandb-summary.json'
    transformer_models = ModelFactory().transformer_model_names
    metrics = {
        'bigearthnet': 'test_mAP_macro',
        'imagenet': 'test_accuracy_macro'
    }

    def __init__(self):
        self.results = self.get_results(save_results=True)

    def get_model_type(self, model_name):
        return 'transformer' if model_name in self.transformer_models else 'cnn'

    def _get_run_details(self, run_dir_name):
        if '--' not in run_dir_name:
            return None
        run_timestamp, run_details_str = run_dir_name.split('--')
        run_name_details = run_details_str.split('-')
        run_results = {
            'timestamp': run_timestamp.replace('run-', ''),
            'dataset': run_name_details[0],
            'model': run_name_details[1],
            'model_type': self.get_model_type(run_name_details[1]),
            'run_type': run_name_details[2],
            'metric': self.metrics[run_name_details[0]],
            'transform': run_name_details[3] if run_name_details[2] != 'ST' else None,
            'transform_param': run_name_details[4] if run_name_details[2] != 'ST' else None,
        }
        try:
            run_results['transform_param'] = int(run_results['transform_param'])
        except:
            try:
                run_results['transform_param'] = float(run_results['transform_param'])
            except:
                pass
        return run_results

    def get_results(self, save_results=False):
        error_runs = []
        all_run_results = []
        for run_dir_name in os.listdir(self.all_runs_dir):
            if not run_dir_name.startswith('run'):
                continue
            try:
                log_file_path = f"{self.all_runs_dir}/{run_dir_name}/{self.internal_log_path}"
                run_results = self._get_run_details(run_dir_name)
                with open(log_file_path, 'r') as file:
                    log_data = json.load(file)
                    run_results['score'] = log_data.get(run_results['metric'], None)
                all_run_results.append(run_results)
            except Exception as e:
                error_runs.append(run_dir_name)

        # self.log.debug(f"Error in processing runs: {', '.join(error_runs)}")

        run_results_df = pd.DataFrame(all_run_results)
        run_results_df = run_results_df.drop_duplicates()
        if save_results:
            run_results_df.to_csv(self.results_df_path, index=False)
        return run_results_df


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = f"{CONFIG['output_dir']}/transform_results"
    results_df_path = f"{output_dir}/results.csv"
    all_transforms = list(TransformFactory().transforms.keys())
    all_datasets = ['bigearthnet', 'imagenet']
    line_styles = {
        'bigearthnet': '-',
        'imagenet': '--'
    }
    marker = {
        'transformer': '^',
        'cnn': 'x'
    }
    linewidth_metric = 0.5

    def __init__(self):
        self.results = ResultsExtractor().get_results()
        self.subplot_fig = None
        self.subplot_ax = None


    def plot_single_results_dataset_avg(self, ax, dataset_results, dataset):
        dataset_avg = dataset_results.groupby('transform_param')
        dataset_avg = dataset_avg.agg({'score': 'mean'}).reset_index()
        line_styles = self.line_styles[dataset]
        ax.plot(
            dataset_avg['transform_param'],
            dataset_avg['score'],
            marker='o',
            linestyle=line_styles,
            label=f"{dataset} Avg",
            color='black',
            zorder=100
        )

    def plot_single_model_results(self, ax, model, dataset_results, line_styles):
        model_results = dataset_results[dataset_results['model'] == model]
        model_type = ResultsExtractor().get_model_type(model)
        x_values = model_results['transform_param']
        y_values = model_results['score']
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        ax.plot(
            x_values,
            y_values,
            marker=self.marker[model_type],
            linestyle=line_styles,
            label=model
        )

    def plot_single_results(self, ax, dataset, results, plot_only_avg, models=None):
        dataset_results = results[results['dataset'] == dataset]
        line_styles = self.line_styles[dataset]
        models = models or dataset_results['model'].unique()
        if not plot_only_avg:
            for model in models:
                try:
                    self.plot_single_model_results(ax, model, dataset_results, line_styles)
                except:
                    pass
            # add line styles for dataset to legend
            ax.plot([], [], linestyle=line_styles, label=dataset, color='black')
        self.plot_single_results_dataset_avg(ax, dataset_results, dataset)
        x_ticks = dataset_results['transform_param'].unique()
        x_ticks = sorted(list(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_yticks([i * 0.1 for i in range(0, 11)])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(f"Model Performances (Acc or mAP macro)")
        for y_value in [i * 0.1 for i in range(1, 10)]:
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=self.linewidth_metric)

    def _clean_input_data(self, datasets, transform_types):
        if datasets is None:
            datasets = datasets or self.all_datasets
        elif isinstance(datasets, str):
            datasets = [datasets]
        if transform_types is None:
            transform_types = self.all_transforms
        elif isinstance(transform_types, str):
            transform_types = [transform_types]
        return datasets, transform_types


    def _get_current_fig_ax(self, subplots, idx, num_plots):
        nrows = 3 if num_plots > 6 else 2 if num_plots > 1 else 1
        ncols = (num_plots // 2) + (num_plots % 2) if num_plots > 1 else 1
        if subplots:
            if self.subplot_fig is None:
                self.subplot_fig, self.subplot_ax = plt.subplots(nrows, ncols, figsize=(20, 11))
            return self.subplot_fig, self.subplot_ax[idx % nrows, idx // nrows]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            return fig, ax


    def plot_results(self, transform_types=None, datasets=None, models=None, subplots=None, plot_only_avg=False):
        datasets, transform_types = self._clean_input_data(datasets, transform_types)
        results_df = self.results
        num_plots = len(transform_types)
        for idx, transform_type in enumerate(transform_types):
            filtered_results = results_df[results_df['transform'] == transform_type]
            fig, ax = self._get_current_fig_ax(subplots, idx, num_plots)
            for dataset in datasets:
                dataset_results = filtered_results[filtered_results['dataset'] == dataset]
                self.plot_single_results(ax=ax, dataset=dataset, results=dataset_results, models=models, plot_only_avg=plot_only_avg)
            ax.set_title(transform_type.upper())
            ax.legend()
            fig.tight_layout()
            if not subplots:
                fig.savefig(f'{self.output_dir}/{transform_type}.png', dpi=200)
                self.log.info(f"Saved plot {transform_type}.png")

        if subplots:
            self.subplot_fig.savefig(f'{self.output_dir}/all_transforms.png', dpi=200)
            self.log.info(f"Saved plot all_transforms.png")



if __name__ == '__main__':
    evaluator = ResultsPlotter()
    evaluator.plot_results(
        # models=['resnet', 'mvit', 'deit', 'densenet'],
        plot_only_avg=True,
        subplots=True,
        transform_types=['bilateral', 'median', 'gaussian', 'grid_shuffle', 'channel_shuffle', 'greyscale']
    )
