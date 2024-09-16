import os
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from utils.config import ROOT_DIR, CONFIG


class ResultEvaluator:
    dir_path = f'{ROOT_DIR}logs/wandb'
    output_dir = f"{CONFIG['output_dir']}/evaluation"
    results_path = f"{output_dir}/results.json"
    line_styles = {
        'bigearthnet': '-',
        'imagenet': '--'
    }

    def _get_run_details(self, folder_name):
        if '--' not in folder_name:
            return None
        run_name = folder_name.split('--')[1]
        run_details = run_name.split('-')
        dataset_name = run_details[0]
        assert dataset_name in ['imagenet', 'bigearthnet']
        model_name = run_details[1]
        run_type = run_details[2]
        if run_type == 'ST' or run_details[3] == 'None':
            return None
        transform_type = run_details[3]
        transform_param = run_details[4]
        try:
            transform_param = int(transform_param)
        except ValueError:
            pass
        return dataset_name, model_name, run_type, transform_type, transform_param

    def get_results(self, log_path='files/wandb-summary.json'):
        results = {}
        for folder_name in os.listdir(self.dir_path):
            folder_path = f"{self.dir_path}/{folder_name}"
            log_file_path = f"{folder_path}/{log_path}"
            details = self._get_run_details(folder_name)
            if details is None or not os.path.exists(log_file_path):
                continue
            dataset_name, model_name, run_type, transform_type, transform_param = details

            with open(log_file_path, 'r') as file:
                log_data = json.load(file)

            test_metric = 'test_mAP_micro' if dataset_name == 'bigearthnet' else 'test_f1_micro'  # TODO: change to acc & macro map
            test_score = log_data.get(test_metric, None)
            test_score = round(test_score, 3) if test_score else None
            if transform_type not in results:
                results[transform_type] = {}
            if dataset_name not in results[transform_type]:
                results[transform_type][dataset_name] = {}
            if model_name not in results[transform_type][dataset_name]:
                results[transform_type][dataset_name][model_name] = []
            score_tuple = (transform_param, test_score)
            results[transform_type][dataset_name][model_name].append(score_tuple)
        with open(self.results_path, 'w') as file:
            json.dump(results, file)
        return results

    def plot_results(self, transform_type, datasets):
        with open(self.results_path, 'r') as file:
            results = json.load(file)

        plt.figure(figsize=(10, 6))
        all_results = results[transform_type]
        for dataset in datasets:
            if dataset not in all_results:
                continue
            results = all_results[dataset]
            for key, data_points in results.items():
                # Sort data points by x value
                data_points = sorted(data_points, key=lambda x: x[0])
                x_values, y_values = zip(*data_points)
                y_values = [np.nan if v is None else v for v in y_values]
                plt.plot(x_values, y_values, marker='o', label=f"{dataset}_{key}", linestyle=self.line_styles[dataset])

        # Add title and labels
        metric = 'mAP' if dataset == 'bigearthnet' else 'F1'
        plt.title(f"Performance on {transform_type} transform")
        plt.xlabel("Transform Parameter")
        plt.ylabel(f"Metric Score")
        plt.xticks(x_values)
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.ylim(bottom=0)
        for y_value in [i * 0.1 for i in range(1, 10)]:
            plt.axhline(y=y_value, color='gray', linestyle='--', linewidth=0.7)
        plt.legend()
        plt.savefig(f'{self.output_dir}/{transform_type}.png', dpi=200)



if __name__ == '__main__':
    evaluator = ResultEvaluator()
    evaluator.get_results()
    for transform_type in ['grid_shuffle', 'low_pass', 'edges']:
        evaluator.plot_results(transform_type, ['imagenet', 'bigearthnet'])
        # for dataset in ['imagenet', 'bigearthnet']:
        #     evaluator.plot_results(transform_type, [dataset])
