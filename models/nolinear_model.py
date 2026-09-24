import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

class NonLinearModel:
    def __init__(self, degree: int = 2, random_state: int = 42):
        self.degree = degree
        self.random_state = random_state
        self.pipeline = None

    def build_pipline(self, num_cols: list[str], cat_cols: list[str]):
        num_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'poly',
                    PolynomialFeatures(degree=self.degree, include_bias=False),
                ),
                ('scaler', StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                (
                    'onehot',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('regressor', Ridge(alpha=10.0, random_state=self.random_state)),
            ]
        )

        return self.pipeline

    def train_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.pipeline is None:
            self.build_pipeline(num_cols, cat_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.pipeline.fit(X_train, y_train)

        return self.pipeline, X_test, y_test

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.pipeline.predict(X_test)
        return {
            'Model': f'Polynomial Regression (Degree {self.degree})',
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R2': r2_score(y_test, y_pred),
        }