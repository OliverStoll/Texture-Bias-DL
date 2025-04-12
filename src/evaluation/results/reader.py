import pandas as pd


class ResultsReader:
    filter_out_transform_params = {
        'ALL': [31],
        'patch_shuffle': [1],
        'patch_rotation': [1],
        'gaussian': [0.5, 2],
    }
    min_performances = {
        'bigearthnet': 0.146761,
        'rgb_bigearthnet': 0.146667,
        'deepglobe': 0.332418,
        'caltech': 0.041328,
        'caltech_120': 0.044320,
        'caltech_ft': 0.055664,
        'imagenet': 0.000969,
    }
    dataset_channels = {
        'bigearthnet': 12,
        'rgb_bigearthnet': 3,
        'deepglobe': 3,
        'caltech': 3,
        'caltech_120': 3,
        'caltech_ft': 3,
        'imagenet': 3,
    }

    def _filter_experiment(self, data: pd.DataFrame, filter_experiment: str) -> pd.DataFrame:
        if filter_experiment == 'combined':
            data = data[data['transform'].str.contains('~')]
            data['transform_param_labels'] = data['transform_param'].astype(str)
            data['transform_param'] = data['transform_param'].apply(
                lambda x: float(x.split('~')[0]))
        if filter_experiment == 'single':
            data = data[~data['transform'].str.contains('~')]
            data['transform_param_labels'] = data['transform_param'].astype(str)
        return data.copy()

    def _clean_unwanted_params(self, data: pd.DataFrame) -> pd.DataFrame:
        data['transform_param'] = data['transform_param'].astype(float)
        for transform_name, param_value in self.filter_out_transform_params.items():
            if transform_name == 'ALL':
                data = data[~data['transform_param'].isin(param_value)]
            else:
                data = data[~((data['transform'] == transform_name) & data['transform_param'].isin(param_value))]
        return data

    def _clean_not_enough_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        not_enough_channels_condition = (
                (data['dataset'] != 'bigearthnet') &
                (data['transform'].str.startswith('Channel', na=False)) &
                (data['transform_param'] > 3)
        )
        return data[~not_enough_channels_condition]

    def _filter_out_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        unique_idx = data.groupby(['dataset', 'model', 'transform', 'transform_param'])['timestamp'].idxmax()
        unique_data = data.loc[unique_idx]
        return unique_data

    def _fix_channel_param(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Only for transformations with channel, divide the transform_param by the number of channels """
        # get all transforms that are Channel Inversion or Channel Shuffle, but not Channel Mean
        channel_data = data[data['transform'].str.startswith('Channel', na=False) & ~data['transform'].str.contains('Channel Mean')]
        for dataset_name, channels in self.dataset_channels.items():
            dataset_data = channel_data[channel_data['dataset'] == dataset_name]
            dataset_data['transform_param'] = dataset_data['transform_param'] / channels
            # round
            dataset_data['transform_param'] = dataset_data['transform_param'].apply(lambda x: round(x, 2))
            data.update(dataset_data)
        return data

    def _calculate_other_score_types(self, results: pd.DataFrame, metric_type: str) -> pd.DataFrame:
        results['score'] = results['score_' + metric_type]
        results = results.sort_values(by=['transform', 'dataset', 'model', 'transform_param'])
        results = results.reset_index(drop=True)
        grouped_scores = results.groupby(['transform', 'dataset', 'model'])['score']
        results['absolute_loss'] = grouped_scores.transform(lambda x: x.iloc[0] - x)
        results['relative_loss'] = grouped_scores.transform(
            lambda x: (x.iloc[0] - x) / x.iloc[0]
        )
        results['relative_score'] = grouped_scores.transform(
            lambda x: x / x.iloc[0]
        )
        results['cleaned_score'] = -1
        for dataset_name, min_performance in self.min_performances.items():
            dataset_results = results[results['dataset'] == dataset_name]
            grouped_scores = dataset_results.groupby(['transform', 'model'])['score']
            dataset_results['cleaned_score'] = grouped_scores.transform(
                lambda x: (x - min_performance) / (x.iloc[0] - min_performance)
            )
            results.update(dataset_results['cleaned_score'])
        return results

    def read_data(
            self,
            data_path: str,
            filter_for_transforms: str = 'single',
            metric_type: str = 'macro',
    ):
        data = pd.read_csv(data_path)
        data = data.dropna()
        # filter out timestamps string lower than 20241216
        data['dt_timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d_%H%M%S')
        data = data[data['dt_timestamp'] >= pd.Timestamp('2024-12-16')]
        # filter out vit for imagenet
        data = data[~((data['model'] == 'vit') & (data['dataset'] == 'imagenet'))]
        data = self._filter_experiment(data, filter_for_transforms)
        data = self._clean_unwanted_params(data)
        data = self._clean_not_enough_channels(data)
        data = self._filter_out_duplicates(data)
        data = self._fix_channel_param(data)
        data = self._calculate_other_score_types(data, metric_type)
        return data
