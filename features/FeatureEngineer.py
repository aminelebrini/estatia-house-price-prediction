class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, data):
        
        data['TotalSF'] = data['GrLivArea'] + data['TotalBsmtSF']

        data['TotalBaths'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]

        data["YearsSinceRemod"] = data["YrSold"] - data["YearRemodAdd"]

        data['IsRemodeled'] = (data['YearBuilt'] != data['YearRemodAdd']).astype(int)

        porch_cols = [
            'WoodDeckSF',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
        ]
        data['TotalPorchSF'] = data[porch_cols].sum(axis=1)

        self.validate_coherence(data)

        return data

    def validate_coherence(self, data):
        if 'TotalSF' in data.columns:
            data = data[data['TotalSF'] >= 0]
        if 'TotalBaths' in data.columns:
            data = data[data['TotalBaths'] >= 0]
        if 'HouseAge' in data.columns:
            data = data[data['HouseAge'] >= 0]
        if 'YearsSinceRemod' in data.columns:
            data = data[data['YearsSinceRemod'] >= 0]
        if 'TotalPorchSF' in data.columns:
            data = data[data['TotalPorchSF'] >= 0]

        return data
