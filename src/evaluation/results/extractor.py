import os
import pandas as pd
import json
from common_utils.config import CONFIG
from common_utils.logger import create_logger


class ResultsExtractor:
    log = create_logger("Transform Evaluation")
    transformer_models = ['vit', 'deit', 'swin', 'cait', 'pvt', 'pit', 'beit', 'convmixer', 'mvit']
    metric_data_split = 'test'
    metrics = {
        'bigearthnet': 'mAP',
        'rgb_bigearthnet': 'mAP',
        'deepglobe': 'mAP',
        'imagenet': 'accuracy',
        'caltech': 'accuracy',
        'caltech_120': 'accuracy',
        'caltech_ft': 'accuracy',
    }

    def __init__(
            self,
            output_dir: str,
            internal_log_path: str = 'files/wandb-summary.json',
            all_runs_dir: str = '/media/storagecube/olivers/logs/logs/wandb'
    ):
        self.output_dir = output_dir
        self.results_df_path = f"{output_dir}/results.csv"
        self.internal_log_path = internal_log_path
        self.all_runs_dir = all_runs_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.debug(f"Starting Results-Extraction. Results will be saved to {self.results_df_path}")


    def get_model_type(self, model_name):
        return 'transformer' if model_name in self.transformer_models else 'cnn'

    def _get_run_details(self, run_dir_name):
        if '--' not in run_dir_name:
            return None
        run_timestamp, run_details_str = run_dir_name.split('--')
        run_name_details = run_details_str.split('-')
        metric_name = f"{self.metric_data_split}_{self.metrics[run_name_details[0]]}"
        run_results = {
            'timestamp': run_timestamp.replace('run-', ''),
            'dataset': run_name_details[0],
            'model': run_name_details[1],
            'model_type': self.get_model_type(run_name_details[1]),
            'run_type': run_name_details[2],
            'metric': f"{metric_name}",
            'transform': run_name_details[3] if run_name_details[2] != 'ST' else None,
            'transform_param': run_name_details[4] if run_name_details[2] != 'ST' else None,
        }

        return run_results

    def get_single_run_results(self, run_dir_name):
        log_file_path = f"{self.all_runs_dir}/{run_dir_name}/{self.internal_log_path}"
        run_results = self._get_run_details(run_dir_name)
        with open(log_file_path, 'r') as file:
            log_data = json.load(file)
            mAP_class_data = {key: value for key, value in log_data.items() if "test_mAP_" in key}
            run_results['score_micro'] = log_data.get(run_results['metric'] + '_micro', None)
            run_results['score_macro'] = log_data.get(run_results['metric'] + '_macro', None)
            run_results['class_scores'] = json.dumps(mAP_class_data)
        return run_results

    def get_results(self, save_results=False):
        error_runs = []
        all_run_results = []
        all_run_dirs = os.listdir(self.all_runs_dir)
        self.log.info(f"Found {len(all_run_dirs)} runs in {self.all_runs_dir}")
        for idx, run_dir_name in enumerate(all_run_dirs):
            if idx % 100 == 0:
                print(idx)
            if 'caltech-120' in run_dir_name:
                continue
            if not run_dir_name.startswith('run'):
                continue
            try:
                single_run_result = self.get_single_run_results(run_dir_name)
                all_run_results.append(single_run_result)
            except FileNotFoundError:
                error_runs.append(run_dir_name)
            except Exception as e:
                self.log.error(f"Error in processing run {run_dir_name}: {e}")
                error_runs.append(run_dir_name)

        # self.log.debug(f"Error in processing runs: {', '.join(error_runs)}")

        run_results_df = pd.DataFrame(all_run_results)
        run_results_df = run_results_df.drop_duplicates()
        # sort by timestamp column
        run_results_df = run_results_df.sort_values(by='timestamp')
        if save_results:
            self.log.debug(f"Saving results to {self.results_df_path}")
            run_results_df.to_csv(self.results_df_path, index=False)
        return run_results_df


if __name__ == '__main__':
    # Example usage
    output_dir = '/home/olivers/master-thesis/output'
    pipeline = ResultsExtractor(output_dir)
    results = pipeline.get_results(save_results=True)