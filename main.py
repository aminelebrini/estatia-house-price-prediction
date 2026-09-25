import os
import matplotlib.pyplot as plt
from rich.console import Console
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from evaluation.ModelEvaluator import ModelEvaluator
from src.DataCleaner import DataCleaner
from src.DataLoader import DataLoader
from src.DataExtracter import DataExtracter
from features.FeatureEngineer import FeatureEngineer
from models.LinearModel import LinearModel
from models.NonLinearModel import NonLinearModel
from models.TreeModel import TreeModel
from notebooks.exploratory_data_analysis import EDA_visualisation

def main():
    console = Console()
    silver_data_path = (
        Path(__file__).parent / "data" / "silver" / "House_Prices_Silver.csv"
    )

    data_extracter = DataExtracter()
    data_house_prices = data_extracter.extract_data()

    if data_house_prices is not None:
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean_data(data_house_prices)
    else:
        print("Error: Failed to extract data.")
        return

    eda = EDA_visualisation(cleaned_data)
    eda.hist_plot()
    eda.scatter_plot()
    eda.box_plot()
    eda.line_plot_year()
    eda.heatmap_plot()


    data_loader = DataLoader()
    data_loader.load_data_silver(cleaned_data)

    feature_engineer = FeatureEngineer()
    data_silver = data_extracter.data_extracter_file(silver_data_path)

    if data_silver is not None:
        engineered_data = feature_engineer.create_features(data_silver)
        data_loader.load_data_gold(engineered_data)

    print("\n" + "=" * 60)
    print("         STARTING MODEL TRAINING           ")
    print("=" * 60 + "\n")

    df = pd.read_csv("data/gold/House_Prices_Gold.csv")
    df = df.dropna(subset=["SalePrice"])

    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    nums_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    linear_model = LinearModel()
    linear_model.build_pipline(nums_cols, cat_cols)
    _, X_test, y_test = linear_model.train_model(X, y)

    print("Linear:", linear_model.evaluate_model(X_test, y_test))

    non_linear_model = NonLinearModel()
    non_linear_model.build_pipline(nums_cols, cat_cols)
    _, X_test, y_test = non_linear_model.train_model(X, y)

    print("Non-Linear:", non_linear_model.evaluate_model(X_test, y_test))

    tree_model = TreeModel()
    tree_model.build_pipeline(nums_cols, cat_cols)
    _, X_test, y_test = tree_model.train_model(X, y)

    print("Random Forest:", tree_model.evaluate_model(X_test, y_test))

    print("\n" + "=" * 60)
    print("         STARTING MODEL EVALUATION           ")
    print("=" * 60 + "\n")

    models = {
        "Linear Model": linear_model.pipeline,
        "Non-Linear Model": non_linear_model.pipeline,
        "Random Forest": tree_model.pipeline,
    }

    param_grids = {
        "Linear Model": {
            'regressor__alpha': [0.1, 1.0, 10.0],
        },
        "Non-Linear Model": {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
            "preprocessor__num__poly__degree": [1, 2, 3],
        },
        "Random Forest": {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [5, 10, 20]
        }
    }

    evaluator = ModelEvaluator(models, param_grids)
    with console.status("[bold green]Evaluating models...") as status:
        results, best_result = evaluator.evaluate_models(X, y, cv=5, scoring="r2")

    for model_name, result in results.items():
        print(
            f"{model_name} - Mean R2: {result['mean']:.4f}, "
            f"Standard deviation: {result['ecart type']:.4f}"
        )

    print("\n" + "=" * 60)
    print("         BEST RESULTS           ")
    print("=" * 60 + "\n")

    print(f"Best Model: {best_result['model_name']}")
    print(f"Best Score: {best_result['best_score']:.4f}")
    print(f"Best Parameters: {best_result['best_params']}")


    conparaison_lm = linear_model.evaluate_model(X_test, y_test)
    comparaison_nlm = non_linear_model.evaluate_model(X_test, y_test)
    comparaison_rf = tree_model.evaluate_model(X_test, y_test)

    comparaison_df = pd.DataFrame(
        {
            "Model": ["Linear Model", "Non-Linear Model", "Random Forest"],
            "MSE": [conparaison_lm['MSE'], comparaison_nlm['MSE'], comparaison_rf['MSE']],
            "MAE": [conparaison_lm['MAE'], comparaison_nlm['MAE'], comparaison_rf['MAE']],
            "RMSE": [conparaison_lm['RMSE'], comparaison_nlm['RMSE'], comparaison_rf['RMSE']],
            "R2 Score": [conparaison_lm['R2'], comparaison_nlm['R2'], comparaison_rf['R2']],
        }
    )

    print("=" * 60 + "\n")
    print("         MODEL COMPARISON RESULTS           ")
    print(comparaison_df)
    print("=" * 60 + "\n")

    predictions = {
        "Linear Model": linear_model.pipeline.predict(X_test),
        "Non-Linear Model": non_linear_model.pipeline.predict(X_test),
        "Random Forest": tree_model.pipeline.predict(X_test),
    }

    for model_name, y_pred in predictions.items():
        errors = y_test - y_pred
        min_price = min(y_test.min(), y_pred.min())
        max_price = max(y_test.max(), y_pred.max())

        # Prix reel vs prix predit
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([min_price, max_price], [min_price, max_price], "r--")
        plt.xlabel("Prix reel")
        plt.ylabel("Prix predit")
        plt.title(f"Prix reel vs prix predit - {model_name}")
        plt.tight_layout()
        plt.show()

        # Distribution des erreurs
        plt.figure(figsize=(7, 5))
        plt.hist(errors, bins=30, edgecolor="black")
        plt.axvline(0, color="red", linestyle="--")
        plt.xlabel("Erreur (prix reel - prix predit)")
        plt.ylabel("Nombre de maisons")
        plt.title(f"Distribution des erreurs - {model_name}")
        plt.tight_layout()
        plt.show()

        # Analyse des residus
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, errors, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Prix predit")
        plt.ylabel("Residu (prix reel - prix predit)")
        plt.title(f"Analyse des residus - {model_name}")
        plt.tight_layout()
        plt.show()

        worst_predictions = pd.DataFrame(
            {
                "Prix reel": y_test,
                "Prix predit": y_pred,
                "Erreur absolue": abs(errors),
            }
        ).sort_values("Erreur absolue", ascending=False)
        print(f"\n5 plus grandes erreurs - {model_name}")
        print(worst_predictions.head())

    # Comparaison des trois modeles avec les metriques principales
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(comparaison_df["Model"], comparaison_df["MAE"])
    axes[0].set_title("Comparaison MAE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(comparaison_df["Model"], comparaison_df["RMSE"])
    axes[1].set_title("Comparaison RMSE")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(comparaison_df["Model"], comparaison_df["R2 Score"])
    axes[2].set_title("Comparaison R2")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
