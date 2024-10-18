"""Training models"""

import argparse

import joblib
import pandas as pd
import yaml

from src.training.train_cross_val import train_cross_val
from src.utils.logs import get_logger


def train(config_path: dict) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path, encoding='utf-8') as conf_file:
        config = yaml.safe_load(conf_file)

    # -----------------------------------------------
    # Read params for the training steps
    # -----------------------------------------------
    target_column = config["dummy_classifier"]["target_variable"]
    group_cv_variable = config["data_split"]["group_cv_variable"]
    estimator_configs = config["train"]["param_grid"]
    cross_validation_n_splits = config["train"]["cross_validation_n_splits"]
    scoring_metric = config["train"]["scoring_metric"]
    model_name = config["train"]["model_name"]

    logger = get_logger("TRAINING_STEP", log_level=config["base"]["log_level"])

    logger.info("Get estimator name")
    logger.info("Estimator: %s", model_name)

    logger.info("Load train dataset")
    train_df = pd.read_csv("./data/processed/train_dataset_fs.csv")
    groups = train_df[group_cv_variable]

    # Split the training data into features (X_train) and target (y_train)
    train_df = train_df.reset_index(drop=True)
    logger.info("cross_validation_n_splits: %s", cross_validation_n_splits)

    logger.info("Train Cross-Validation to get best params")
    scores, model, cross_val_pred_train_df = train_cross_val(
        train_df=train_df,
        target_column=target_column,
        group_cv_variable=group_cv_variable,
        estimator_configs=estimator_configs,  # filtered_estimator_configs,
        cross_validation_n_splits=cross_validation_n_splits,
        groups=groups,
    )

    logger.info("Best score: %s", model)

    cross_val_scores_df = pd.DataFrame(scores)

    cv_accuracy = round(cross_val_scores_df.test_accuracy_score.mean(), 3)
    cv_precision = round(cross_val_scores_df.test_precision_score.mean(), 3)

    logger.info("CV ACCURACY: %s", cv_accuracy)
    logger.info("CV PRECISION: %s", cv_precision)

    cross_val_pred_train_df.to_csv("./models/cross_val_pred.csv", index=False)
    cross_val_scores_df.to_csv("./models/cross_val_score.csv", index=False)

    logger.info("Save model")
    joblib.dump(model, "./models/model.joblib")

    logger.info("Train Step Complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    train(
        config_path=args.config,
    )
