import pandas as pd
from sklearn.preprocessing import StandardScaler
class DataCleaner:
    def __init__(self):
        
        self.continuous_cols = []
        self.scaler = StandardScaler()


    def clean_data(self, data_house_prices):

        df = data_house_prices.copy()

        na_count = df.isna()
        check_duplicates = df.duplicated().sum()
        print(df.columns)
        print(df.head())
        print(df.describe())
        print(df.info())
        max_sales_price = df['SalePrice'].max()
        min_sales_price = df['SalePrice'].min()
        max_grlivarea = df['GrLivArea'].max()
        min_grlivarea = df['GrLivArea'].min()
        # print(f"Max sales price: {max_sales_price}")
        # print(f"Min sales price: {min_sales_price}")
        # print(f"Max GrLivArea: {max_grlivarea}")
        # print(f"Min GrLivArea: {min_grlivarea}")
        if check_duplicates > 0:
            df = df.drop_duplicates()
        if 'SalePrice' in df.columns:
            df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
        else:
            is_outliers = df[df['GrLivArea'] > 4000]
            df = df[~is_outliers]

        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(cat_cols) > 0:
            df.loc[:,cat_cols] = df[cat_cols].fillna('None')
        if len(num_cols) > 0:
            df.loc[:, num_cols] = df[num_cols].fillna(0)

        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
        for col in ordinal_cols:
            if col in df.columns:
                df.loc[col] = df[col].map(qual_map)

        continuous_cols = []
        all_nums_col = df.select_dtypes(include=['int64', 'float64']).columns
        on_continuous_cols = [
            'Id', 'MSSubClass', 'OverallQual', 'OverallCond', 
            'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
            'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
            'MoSold', 'YrSold', 'SalePrice'
        ]
        for col in all_nums_col:
            if col not in on_continuous_cols:
                continuous_cols.append(col)

        self.continuous_cols = continuous_cols

        return df

    def x_y_separation(self, df):

        best_features_x = [
            'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'LotArea',
    
            'OverallQual', 'OverallCond', 'ExterQual', 'Functional',
    
            'Neighborhood', 'MSZoning', 'HouseStyle', 'Foundation',
    
            'YearBuilt', 'YearRemodAdd',
    
            'FullBath', 'HalfBath', 'BsmtFullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenAbvGr',
    
            'GarageCars', 'GarageArea', 'GarageType', 'GarageFinish', 'MasVnrArea'
        ]
        x = df.drop(['Id', 'SalePrice'], axis=1)
        categorical_cols = x.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)
        y = df['SalePrice']

        self.X_train = x
        self.Y_train = y

    def standarisation_normalisation(self, X_train, X_test):
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        for col in self.continuous_cols:
            if col in X_train_scaled.columns:
                X_train_scaled[col] = self.scaler.fit_transform(X_train_scaled[[col]])

        return X_train_scaled, X_test_scaled