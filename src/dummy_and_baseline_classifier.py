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
    target_column = config["dummy_classifier"]["target_variable"]

    logger.info("Load train dataset")
    train_df = pd.read_csv("./data/processed/train_dataset.csv")
    test_df = pd.read_csv("./data/processed/test_dataset.csv")

    logger.info("Multiple Models Pre Train:")

    x_train = train_df.drop([target_column], axis=1)
    y_train = train_df[target_column]

    x_test = test_df.drop([target_column], axis=1)
    y_test = test_df[target_column]

    # --------------------------------
    # Dummy Classifier

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_train, y_train)
    DummyClassifier(strategy="most_frequent")
    dummy_clf.predict(x_test)
    dummy_classifier_score = round(dummy_clf.score(x_test, y_test), 3)

    logger.info(f"Dummy Classifier Score: %s", dummy_classifier_score)

    # --------------------------------
    # Baseline Classifier - Simple Domain expert rules

    test_df["benchmark_prob"] = np.where(
        test_df["before_average_W_ratio"] > test_df["before_average_W_ratio_opp"], 1, 0
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        test_df["results"], test_df["benchmark_prob"]
    ).ravel()
    precision_metric = round(true_positive / (true_positive + false_positive), 3)
    specificity_metric = round(true_negative / (true_negative + false_positive), 3)
    recall_metric = round(true_positive / (true_positive + false_negative), 3)
    f1_metric = round(
        (2 * precision_metric * recall_metric) / (precision_metric + recall_metric), 3
    )
    accuracy_metric = round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative), 3)

    logger.info("Baseline Accuracy: %s", accuracy_metric)
    logger.info("Baseline Precision: %s", precision_metric)
    logger.info("Baseline Specificity: %s", specificity_metric)
    logger.info("Baseline Recall: %s", recall_metric)
    logger.info("Baseline F1: %s", f1_metric)

    report = {
        "baseline_accuracy": round(accuracy_metric, 3),
        "baseline_precision": round(precision_metric, 3),
        "baseline_specificity": round(specificity_metric, 3),
        "baseline_recall": round(recall_metric, 3),
        "baseline_f1": round(f1_metric, 3),
    }

    logger.info("Save metrics")

    with open(
        "data/reports/baseline_classifier_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            obj={
                "baseline_accuracy": report["baseline_accuracy"],
                "baseline_precision": report["baseline_precision"],
                "baseline_specificity": report["baseline_specificity"],
                "baseline_recall": report["baseline_recall"],
                "baseline_f1": report["baseline_f1"],
            },
            fp=file,
        )


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    dummy_and_baseline_classifier(config_path=args.config)
