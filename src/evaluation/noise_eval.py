from plotting import ResultsReader


reader = ResultsReader()
data = reader.read_data(
    data_path='C:/CODE/master-thesis/data/results_v4.csv',
    filter_for_transforms='single',
)
data = data[data['transform'] == 'noise']

# calculate mean and std for each dataset
for dataset in data['dataset'].unique():
    dataset_data = data[data['dataset'] == dataset]
    mean_data = dataset_data.groupby('transform_param').agg({'score': 'mean'}).reset_index()
    # print 5th relative_score for each dataset
    print(f"{dataset}: {mean_data['score'].iloc[5]:.3f}")
    # print(f"{mean_data['score']}")