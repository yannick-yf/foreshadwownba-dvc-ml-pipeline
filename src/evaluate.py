"""Pre Train Multiple models."""

# https://machinelearningmastery.com/multi-output-regression-models-with-python/

import os
from pathlib import Path
import pandas as pd
import boto3

# Machine Learning package
from sklearn.model_selection import GroupKFold

from typing import Text
import yaml
import argparse
from decouple import config
import os

# from src.stages.visualization import plot_regression_pred_actual
from src.utils.logs import get_logger
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

import json
import joblib

import shap
import matplotlib.pyplot as plt

import pandas as pd

# import shap
import matplotlib.pyplot as plt

logger = get_logger("EVALUATION_STEP", log_level="INFO")

def write_plot_regression_data(y_true, predicted, filename):
    """
    Write the true and predicted values of a regression model to a CSV file.

    Args:
        y_true (list): The true values of the target variable.
        predicted (list): The predicted values of the target variable.
        filename (str): The name of the CSV file to write the data to.
    """
    assert len(predicted) == len(y_true)
    reg_plot = pd.DataFrame(
        list(zip(y_true, predicted)), columns=["y_true", "predicted"]
    )
    reg_plot.to_csv(filename, index=False)

def evaluate(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)

    #-----------------------------------------------
    # Read daa for feature creation

    logger = get_logger(
        'EVALUATE', 
        log_level=config['base']['log_level'])
    
    #-----------------------------------------------
    # Read input params

    target_column = config['dummy_classifier']['target_variable']
    group_cv_variable = config['data_split']['group_cv_variable']
    metrics_path = config['evaluate']['metrics_path']

    logger.info("Load trained model")
    model = joblib.load('./models/model.joblib')

    logger.info("Load Cross Val Scores")
    cross_val_scores_df = pd.read_csv('./models/cross_val_score.csv')

    logger.info("Evaluate CV scores DataFrame")
    cv_accuracy = round(cross_val_scores_df.test_accuracy_score.mean(),3)
    cv_precision = round(cross_val_scores_df.test_precision_score.mean(),3)

    logger.info("Load test dataset")

    train_df = pd.read_csv('./data/processed/train_dataset_fs.csv')
    test_df = pd.read_csv('./data/processed/test_dataset_fs.csv')

    list_columns_to_delete = [
        target_column, 
        group_cv_variable,
        'id_season',	
        'tm',	
        'opp',
        ]

    logger.info("Evaluate (build report)")
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(list_columns_to_delete, axis=1).values

    prediction = model.predict(X_test)

    evaluation_precision = precision_score(y_test, prediction)
    evaluation_accuracy = accuracy_score(y_test, prediction)

    logger.info(f"EVALUATION PRECISION: {round(evaluation_precision, 3)};")
    logger.info(f"EVALUATION ACCURACY: {round(evaluation_accuracy, 3)};")

    report = {
        "cv_accuracy": round(cv_accuracy, 3),
        "cv_precision": round(cv_precision, 3),
        "evaluation_accuracy": round(evaluation_accuracy, 3),
        "evaluation_precision": round(evaluation_precision, 3),
        "actual": y_test,
        "predicted": prediction,
    }

    logger.info("Save metrics")

    with open('data/reports/metrics.json', "w", encoding="utf-8") as fp:
        json.dump(
            obj={
                "accuracy_cv_score": report["cv_accuracy"],
                "precision_cv_score": report["cv_precision"],
                "accuracy_evaluation_score": report["evaluation_accuracy"],
                "precision_evaluation_score": report["evaluation_precision"],
            },
            fp=fp,
        )

    logger.info(f"Accuracy & Precision metrics file saved to : {'data/reports/metrics.json'}")

    # Save shape value plot
    # Shape value framework
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(train_df.drop(list_columns_to_delete, axis=1))

    plt.figure(figsize=(10, 10))
    shap_beeswarm_path = './data/reports/shap_beeswarm.png'
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(shap_beeswarm_path, bbox_inches="tight", dpi=100)

    plt.figure(figsize=(10, 10))
    shap.plots.bar(shap_values, show=False)
    plt.savefig('./data/reports/shap_plot_bar.png', bbox_inches="tight", dpi=100)

    logger.info(
        f"Shap plots saved to : {shap_beeswarm_path, './data/reports/shap_plot_bar.png' }"
    )

    # regression_data_path = './data/reports/regression_plot_data.csv'
    # write_plot_regression_data(y_test, prediction, filename=regression_data_path)

    logger.info("Evaluate Step Done")

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--config', 
        dest='config', 
        required=True)
    
    args = arg_parser.parse_args()

    evaluate(
        config_path = args.config
        )