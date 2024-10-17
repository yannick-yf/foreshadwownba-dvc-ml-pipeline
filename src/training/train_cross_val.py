"""Train model with cross validation."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from xgboost import XGBClassifier

class UnsupportedClassifier(Exception):
    """Exception raised for unsupported classifiers.

    Attributes:
        estimator_name -- name of the unsupported estimator
    """

    def __init__(self, estimator_name):
        self.msg = f"Unsupported estimator {estimator_name}"
        super().__init__(self.msg)


def get_estimator(estimator_name, **kwargs):
    """
    Returns an instance of the estimator based on the provided name and hyperparameters.
    """
    if estimator_name == "Random Forest Regressor":
        estimator = RandomForestClassifier(**kwargs)
    elif estimator_name == "Extreme Gradient Boosting":
        estimator = XGBClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    return estimator


def get_supported_estimator() -> dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        "Random Forest Classifier": RandomForestClassifier,
        "Extreme Gradient Boosting": XGBClassifier,
    }


def train_cross_val(
    train_df: pd.DataFrame,
    target_column: str,
    group_cv_variable: str,
    estimator_configs: list[dict[str, any]],
    # param_grids: dict,
    cross_validation_n_splits: int,
    groups: pd.Series,
    scoring_metric: str = "r2",
) -> tuple[str, pd.DataFrame]:
    """Train model.
    Args:
        df {pandas.DataFrame}: dataset
        target_column {Text}: target column name
        param_grids {Dict}: dictionary containing grid parameters for each estimator
        cv {int}: cross-validation value
    Returns:
        best_estimator, best_scores
    """

    # estimators = get_supported_estimator()
    groups = groups
    X_train = train_df.drop(
        [
            target_column,
            group_cv_variable,
            "id_season",
            "tm",
            "opp",
        ],
        axis=1,
    )

    y_train = train_df[target_column]
    cv = GroupKFold(n_splits=cross_validation_n_splits)

    scoring = {
        "precision_score": make_scorer(precision_score),
        "accuracy_score": make_scorer(accuracy_score),
    }

    estimator = XGBClassifier(**estimator_configs)

    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        groups=groups,
        return_estimator=True,
        return_train_score=True,
        n_jobs=-1,
    )

    train_w_cv_pred = get_predicted_values_from_cross_val(
        target_column,
        group_cv_variable,
        estimator,
        train_df,
        X_train,
        y_train,
        cv,
        groups,
    )

    estimator.fit(X_train, y_train)

    return scores, estimator, train_w_cv_pred


def get_predicted_values_from_cross_val(
    target_column, group_cv_variable, estimator, train_df, X_train, y_train, cv, groups
) -> pd.DataFrame:
    """Get predicted values from cross validation."""

    y_cv_pred_proba = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        n_jobs=-1,
        method="predict_proba",
    )

    y_cv_pred_results = cross_val_predict(
        estimator, X_train, y_train, cv=cv, groups=groups, n_jobs=-1, method="predict"
    )

    train_w_cv_pred = train_df[
        [
            target_column,
            group_cv_variable,
            "id_season",
            "tm",
            "opp",
        ]
    ]

    y_cv_pred_proba_df = pd.DataFrame(y_cv_pred_proba)
    y_cv_pred_proba_df.columns = ["y_cv_pred_0", "y_cv_pred_1"]

    train_w_cv_pred["y_cv_pred_0"] = y_cv_pred_proba_df["y_cv_pred_0"].copy()
    train_w_cv_pred["y_cv_pred_1"] = y_cv_pred_proba_df["y_cv_pred_1"].copy()
    train_w_cv_pred["y_cv_pred_results"] = y_cv_pred_results

    return train_w_cv_pred
