from evaluation.latex.constants import dataset_names_friendly


def prepare_data(reader_data):
    data = []
    # filter out convnext for caltech and caltech_120
    reader_data = reader_data[~((reader_data['dataset'] == 'caltech') & (reader_data['model'] == 'convnext'))]
    reader_data = reader_data[~((reader_data['dataset'] == 'caltech_120') & (reader_data['model'] == 'convnext'))]
    # Assuming your DataFrame is named df
    max_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmax()]
    # min_intensity = reader_data.loc[reader_data.groupby(['dataset', 'model', 'transform'])['transform_param'].idxmin()]
    data = max_intensity
    # Group by dataset and transform, then calculate the average score across models
    scores = max_intensity.groupby(['dataset', 'transform'])['score'].mean().reset_index()
    cleaned_scores = max_intensity.groupby(['dataset', 'transform'])['cleaned_score'].mean().reset_index()
    dataset_scores = cleaned_scores.sort_values(by='transform')
    best_model = data.loc[data.groupby(["dataset", "transform"])["cleaned_score"].idxmax()][
        ['dataset', 'transform', 'model', 'cleaned_score', 'score']]
    avg_model = data.groupby(["dataset", "transform"], as_index=False)["cleaned_score"].mean()
    # replace cleaned_score with best_score and avg_score
    best_model = best_model.rename(columns={"cleaned_score": "Best Relative Performance"})
    avg_model = avg_model.rename(columns={"cleaned_score": "Avg. Relative Performance"})
    data = best_model.merge(avg_model, on=["dataset", "transform"])
    data = data.rename(columns={"dataset": "Dataset", "transform": "Transformation", "model": "Model"})
    # add domain column either cv or rs
    rs_datasets = ['BigEarthNet', 'DeepGlobe', 'RGB BigEarthNet']
    data['Dataset'] = data['Dataset'].apply(lambda x: dataset_names_friendly[x])
    data['domain'] = data['Dataset'].apply(lambda x: 'rs' if x in rs_datasets else 'cv')
    # sort by dataset domain
    data = data.sort_values(by=['domain'], ascending=False)
    return data
