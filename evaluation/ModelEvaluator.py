from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_validate


class ModelEvaluator:
    def __init__(self, models, param_grids):
        self.param_grids = param_grids
        self.models = models

    def evaluate_models(self, X, y, cv, scoring):
        results = {}
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            scores = cross_val_score(
                model,
                X,
                y,
                cv=kf,
                scoring=scoring,
            )

            scores_validate = cross_validate(
                model,
                X,
                y,
                cv=kf,
                scoring=scoring,
            )
            results[name] = {
                "scores": scores.max(),
                "ecart type": scores.std(),
                "cv": kf,
                "scores_validate": scores_validate,
                "mean": scores.mean(),
            }

        grid_search = GridSearchCV(
            self.models["Linear Model"],
            param_grid=self.param_grids["Linear Model"],
            cv=kf,
            scoring=scoring,
        )
        grid_search.fit(X, y)

        best_result = {
            "model_name": "Linear Model",
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "best_model": grid_search.best_estimator_,
        }

        return results, best_result
