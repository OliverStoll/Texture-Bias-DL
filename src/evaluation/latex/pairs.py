from plotting import ResultsReader
import warnings
import pandas as pd

from evaluation.latex.formatting import get_best_avg_table
from evaluation.results.data import prepare_data

warnings.filterwarnings("ignore")


def formal_table_data(table: pd.DataFrame) -> str:
    transform_to_category = {
        'Bilateral~Channel Shuffle': 'Shape',
        'Bilateral~Patch Shuffle': 'Spectral',
        'Patch Shuffle~Channel Shuffle': 'Texture'
    }
    table = table.rename(columns=transform_to_category)
    table = table[['Dataset', 'Spectral', 'Texture', 'Shape', 'Sum']]
    table_str = table.to_latex(index=False, float_format="%.2f")
    return table_str


reader_data = ResultsReader().read_data(
    data_path='/data/run_results.csv',
    filter_for_transforms='combined',
)
results_data = prepare_data(reader_data)
avg_table, best_table = get_best_avg_table(results_data)







