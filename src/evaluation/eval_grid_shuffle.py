import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.config import ROOT_DIR, CONFIG


def get_grid_shuffle_results(dir_path, save_path, log_path='files/wandb-summary.json'):
    results = {}
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)

        # Check if the current item is a directory
        if not os.path.isdir(folder_path) or '--' not in folder_name:
            continue

        run_name = folder_name.split('--')[1]
        run_details = run_name.split('-')
        model_name = run_details[0]
        dataset_name = run_details[1]
        grid = int(run_details[3]) if len(run_details) > 3 else int(run_details[2])
        log_file = os.path.join(folder_path, log_path)
        with open(log_file, 'r') as file:
            log_data = json.load(file)

        test_f1_score = log_data.get('test_f1', None)
        test_f1_score = round(test_f1_score, 3) if test_f1_score else None
        if dataset_name not in results:
            results[dataset_name] = {}
        if model_name not in results[dataset_name]:
            results[dataset_name][model_name] = [None] * 20
        results[dataset_name][model_name][grid-1] = test_f1_score

    with open(save_path, 'w') as file:
        json.dump(results, file)

    return results


def plot_grid_shuffle_results(results_path, dataset='imagenet'):
    with open(results_path, 'r') as file:
        results = json.load(file)

    results = results[dataset]

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Plot each series
    for key, values in results.items():
        y_values = [np.nan if v is None else v for v in values]
        x_values = list(range(1, len(values) + 1))
        plt.plot(x_values, y_values, marker='o', label=key)

    # Add title and labels
    plt.title("Performance on grid-shuffled test set")
    plt.xlabel("Grid-Shuffle Size")
    plt.ylabel("F1 Score")
    plt.xticks(range(1, 21, 1))
    plt.yticks([i * 0.1 for i in range(0, 11)])
    plt.ylim(bottom=0)
    for y_value in [i * 0.1 for i in range(1, 10)]:
        plt.axhline(y=y_value, color='gray', linestyle='--', linewidth=0.7)

    plt.legend()
    plt.savefig('grid_shuffle_plot.png', dpi=200)




if __name__ == '__main__':
    grid_shuffle_results_path = f"{CONFIG['output_dir']}/grid_shuffle/results.json"
    grid_shuffle_results = get_grid_shuffle_results(dir_path=f'{ROOT_DIR}logs/wandb',
                                                    save_path=grid_shuffle_results_path)
    print(grid_shuffle_results)
    plot_grid_shuffle_results(grid_shuffle_results_path)
