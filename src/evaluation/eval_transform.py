import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from common_utils.config import ROOT_DIR, CONFIG
from common_utils.logger import create_logger

from models import ModelFactory
from transforms import TransformFactory
from datasets import DataLoaderFactory

from evaluation.results_pipeline import ResultsExtractor


class ResultsPlotter:
    log = create_logger("Transform Plotter")
    output_dir = f"{CONFIG['output_dir']}/transform_results"
    results_df_path = f"{output_dir}/results.csv"
    all_transforms = list(TransformFactory().transforms.keys())
    all_datasets = list(DataLoaderFactory().dataset_names)
    line_styles = {
        'bigearthnet': '-',
        'imagenet': '--'
    }
    marker = {
        'transformer': '^',
        'cnn': 'x'
    }
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linewidth_metric = 0.5

    def __init__(self, results):
        self.results = results
        self.subplot_fig = None
        self.subplot_ax = None
        self.color_idx = 0


    def plot_single_results_dataset_avg(self, ax, dataset_results, dataset, color=None):
        dataset_avg = dataset_results.groupby('transform_param')
        dataset_avg = dataset_avg.agg({'score': 'mean'}).reset_index()
        ax.plot(
            dataset_avg['transform_param'],
            dataset_avg['score'],
            marker='o',
            linestyle=self.line_styles.get(dataset, '-'),
            label=f"{dataset} Avg",
            color='black' if color is None else color,
            zorder=100
        )

    def plot_single_model_results(self, ax, model, dataset_results, line_styles, color=None):
        model_results = dataset_results[dataset_results['model'] == model]
        model_type = ResultsExtractor().get_model_type(model)
        x_values = model_results['transform_param']
        y_values = model_results['score']
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        ax.plot(
            x_values,
            y_values,
            marker=self.marker.get(model_type, 'o'),
            linestyle=line_styles,
            label=model,
            color=color,
        )

    def plot_single_results(self, ax, dataset, results, plot_only_avg, models=None):
        dataset_results = results[results['dataset'] == dataset]
        line_styles = self.line_styles.get(dataset, '-')
        models = models or dataset_results['model'].unique()
        if not plot_only_avg:
            for model in models:
                try:
                    self.plot_single_model_results(ax, model, dataset_results, line_styles)
                except:
                    pass
            avg_color = 'black'
        else:
                if self.col
            avg_color = self.colors[self.color_idx]
            self.color_idx += 1
        ax.plot([], [], linestyle=line_styles, label=dataset, color=avg_color)
        self.plot_single_results_dataset_avg(ax, dataset_results, dataset, color=avg_color)
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
        self.log.info(f"Datasets: {datasets}")
        self.log.info(f"Transforms: {transform_types}")
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
                save_dir = f"{self.output_dir}/{transform_type}.png"
                fig.savefig(save_dir, dpi=200)
                self.log.info(f"Saved plot {save_dir}")

        if subplots:
            save_dir = f"{self.output_dir}/all_transforms.png"
            self.subplot_fig.savefig(save_dir, dpi=200)
            self.log.info(f"Saved plot {save_dir}")



if __name__ == '__main__':
    results = ResultsExtractor().get_results()
    evaluator = ResultsPlotter(results=results)
    evaluator.plot_results(
        # models=['resnet', 'mvit', 'deit', 'densenet'],
        plot_only_avg=True,
        subplots=True,
        # transform_types=['bilateral', 'median', 'gaussian', 'grid_shuffle',
        # 'channel_shuffle', 'greyscale']
    )
