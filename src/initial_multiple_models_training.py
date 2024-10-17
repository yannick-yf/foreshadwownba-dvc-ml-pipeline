"""Pre Train Multiple models."""

import argparse
from typing import Text

import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from pycaret.classification import *

from src.utils.logs import get_logger

logger = get_logger("PRE_TRAIN_MULTIPLE_MODELS", log_level="INFO")

def pre_train_multplie_models(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # -----------------------------------------------
    # Read data for feature creation

    logger = get_logger(
        "INITIAL_MULTIPLE_MODELS_TRAINING", log_level=config["base"]["log_level"]
    )

    # -----------------------------------------------
    # Read input params
    target_column = config["dummy_classifier"]["target_variable"]
    random_state = config["base"]["random_state"]
    group_cv_variable = config["data_split"]["group_cv_variable"]
    cross_validation_n_splits = config["initial_multiple_models_training"][
        "cross_validation_n_splits"
    ]

    logger.info("Load train dataset")
    train_df = pd.read_csv("./data/processed/train_dataset_fs.csv")

    logger.info("Multiple Models Pre Train:")

    groups = train_df[group_cv_variable]

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

    classification_exp = ClassificationExperiment()

    # init setup on exp
    classification_exp.setup(
        X_train,
        target=y_train,
        session_id=random_state,
        fold_strategy=GroupKFold(n_splits=cross_validation_n_splits),
        fold_groups=groups,
        n_jobs=-1,  # Use all available CPU cores
        # early_stopping=True,  # Early stopping to prevent overfitting
        log_experiment=False,  # Log experiment for tracking
        experiment_name="initial_model_training",  # Experiment name for logging
    )

    # compare baseline models
    classification_exp.compare_models()
    results_df = classification_exp.pull()

    results_df["Precision"] = results_df["Prec."].round(2)
    results_df["Accuracy"] = results_df["Accuracy"].round(2)

    accuracy_metric_by_model = results_df.sort_values(
        by="Accuracy", ascending=False
    ).head(3)[["Model"]]

    accuracy_metric_by_model.to_csv("./data/processed/top_3_models.csv", index=False)

    precision_metric_by_model = results_df.sort_values(
        by="Precision", ascending=False
    ).head(3)[["Model", "Precision"]]

    results_df.to_csv("./data/processed/models_results.csv", index=False)

    logger.info(
        "Top 3 models to test based on Accuracy: "
        + accuracy_metric_by_model["Model"].values[0]
        + " - "
        + accuracy_metric_by_model["Model"].values[1]
        + " - "
        + accuracy_metric_by_model["Model"].values[2]
    )

    logger.info(
        "Top 3 models to test based on Precision: "
        + precision_metric_by_model["Model"].values[0]
        + " - "
        + precision_metric_by_model["Model"].values[1]
        + " - "
        + precision_metric_by_model["Model"].values[2]
    )


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    pre_train_multplie_models(config_path=args.config)
