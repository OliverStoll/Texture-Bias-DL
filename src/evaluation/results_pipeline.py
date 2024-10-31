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


class ResultsExtractor:
    log = create_logger("Transform Evaluation")
    all_runs_dir = f'/media/storagecube/olivers/logs/logs/wandb'
    output_dir = f"{CONFIG['output_dir']}/transform_results"
    results_df_path = f"{output_dir}/results.csv"
    internal_log_path = 'files/wandb-summary.json'
    transformer_models = ModelFactory().transformer_model_names
    metrics = {
        'bigearthnet': 'test_mAP_macro',
        'rgb_bigearthnet': 'test_mAP_macro',
        'deepglobe': 'test_mAP_macro',
        'imagenet': 'test_accuracy_macro',
        'caltech': 'test_accuracy_macro',
        'caltech_120': 'test_accuracy_macro'
    }

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
            if 'caltech-120' in run_dir_name:
                continue
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
        # sort by timestamp column
        run_results_df = run_results_df.sort_values(by='timestamp')
        if save_results:
            run_results_df.to_csv(self.results_df_path, index=False)
        return run_results_df


if __name__ == '__main__':
    pipeline = ResultsExtractor()
    results = pipeline.get_results(save_results=True)