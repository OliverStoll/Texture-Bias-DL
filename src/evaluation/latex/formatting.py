import pandas as pd


def get_best_avg_table(results_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_best = []
    all_avg = []
    for dataset in results_data['Dataset'].unique():
        dataset_best = {'Dataset': dataset}
        dataset_avgs = {'Dataset': dataset}
        d_data = results_data[results_data['Dataset'] == dataset]
        for transforms in ['Bilateral~Channel Shuffle', 'Bilateral~Patch Shuffle', 'Patch Shuffle~Channel Shuffle']:
            t_data = d_data[d_data['Transformation'] == transforms]
            dataset_best[transforms] = t_data['Best Relative Performance'].values[0]
            dataset_avgs[transforms] = t_data['Avg. Relative Performance'].values[0]
        all_best.append(dataset_best)
        all_avg.append(dataset_avgs)
    best_table = pd.DataFrame(all_best)
    avg_table = pd.DataFrame(all_avg)
    best_table = best_table.set_index('Dataset')
    avg_table = avg_table.set_index('Dataset')
    best_table['Sum'] = best_table.sum(axis=1)
    avg_table['Sum'] = avg_table.sum(axis=1)
    avg_table = avg_table.reset_index().rename(columns={"index": "Dataset"})
    best_table = best_table.reset_index().rename(columns={"index": "Dataset"})
    return avg_table, best_table
