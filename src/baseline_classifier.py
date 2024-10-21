"""Dummy Classifier"""


import argparse
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from src.utils.logs import get_logger


def dummy_and_baseline_classifier(config_path: dict) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("DUMMY_CLASSIFIER_STEP", log_level=config["base"]["log_level"])

    # -----------------------------------------------
    # Read input params

    logger.info("Load train dataset")
    train_df = pd.read_csv("./data/processed/test_dataset.csv")
    test_df = pd.read_csv("./data/processed/test_dataset.csv")

    logger.info("Multiple Models Pre Train:")

    # --------------------------------
    # Baseline Classifier - Simple Domain expert rules

    train_df["benchmark_prob"] = np.where(
        train_df["before_average_W_ratio"] > train_df["before_average_W_ratio_opp"], 1, 0
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        train_df["results"], train_df["benchmark_prob"]
    ).ravel()

    train_accuracy_metric = round(
        (true_positive + true_negative)
        / (true_positive + true_negative + false_positive + false_negative),
        3,
    )

    logger.info("Train Baseline Accuracy: %s", train_accuracy_metric)

    test_df["benchmark_prob"] = np.where(
        test_df["before_average_W_ratio"] > test_df["before_average_W_ratio_opp"], 1, 0
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        test_df["results"], test_df["benchmark_prob"]
    ).ravel()

    test_accuracy_metric = round(
        (true_positive + true_negative)
        / (true_positive + true_negative + false_positive + false_negative),
        3,
    )

    logger.info("Test Baseline Accuracy: %s", test_accuracy_metric)

    report = {
        "train_baseline_accuracy": round(train_accuracy_metric, 3),
        "test_baseline_accuracy": round(test_accuracy_metric, 3)
        }

    logger.info("Save metrics")

    with open(
        "data/reports/baseline_classifier_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            obj={
                "train_baseline_accuracy": report["train_baseline_accuracy"],
                "test_baseline_accuracy": report["test_baseline_accuracy"]
                },
            fp=file,
        )


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    dummy_and_baseline_classifier(config_path=args.config)
