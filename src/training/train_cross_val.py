
"""Train model with cross validation."""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    classification_report,
    precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    roc_curve,
    auc)

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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
    if estimator_name == 'Random Forest Regressor':
        estimator = RandomForestClassifier(**kwargs)
    elif estimator_name == 'Extreme Gradient Boosting':
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
        "Extreme Gradient Boosting": XGBClassifier
        }

def train_cross_val(
    train_df: pd.DataFrame,
    target_column: str,
    group_cv_variable: str,
    estimator_configs: list[dict[str,any]],
    #param_grids: dict,
    cross_validation_n_splits: int,
    groups: pd.Series,
    scoring_metric: str = "r2",
    
)-> tuple[str, pd.DataFrame]:
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
    X_train = train_df.drop([
        target_column, 
        group_cv_variable,
        'id_season',	
        'tm',	
        'opp',
        ], axis=1)
    
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

    # ------------------ Train the general models--------------------#
    # Train the model with the best params - for now only one set of params
    estimator.fit(X_train, y_train)


    return scores, estimator

