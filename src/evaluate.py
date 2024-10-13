"""Pre Train Multiple models."""

# https://machinelearningmastery.com/multi-output-regression-models-with-python/

import os
from pathlib import Path
import pandas as pd
import numpy as np
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
    precision_score,
    accuracy_score)

import json
import joblib

import shap
import matplotlib.pyplot as plt

import pandas as pd

# import shap
import matplotlib.pyplot as plt

logger = get_logger("EVALUATION_STEP", log_level="INFO")

def rename_opponent_columns(training_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    """
    training_df.columns = training_df.columns.str.replace('_y', '_opp')
    training_df.columns = training_df.columns.str.replace('_x', '')

    return training_df

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

    prediction_value = model.predict(X_test)

    #----------------------------------------------------
    # Get the proba to get the Metric evaluation per game
    # We will comapre proba to win from team 1 vs team 2
    # It leads to have 1 lign per game
    prediction_proba = model.predict_proba(X_test)
    prediction_proba_df = pd.DataFrame(prediction_proba)
    prediction_proba_df.columns = ['prediction_proba_df_0', 'prediction_proba_df_1']
    
    test_df_w_pred = test_df[[
        target_column, 
        group_cv_variable,
        'id_season',	
        'tm',	
        'opp',
        ]]
    
    test_df_w_pred['prediction_proba_df_0'] = prediction_proba_df['prediction_proba_df_0'].copy()
    test_df_w_pred['prediction_proba_df_1'] = prediction_proba_df['prediction_proba_df_1'].copy()
    test_df_w_pred['prediction_value'] = prediction_value

    test_df_w_pred.to_csv('./models/test_df_w_pred.csv', index=False)

    test_df_w_pred = test_df_w_pred.rename(columns={
        "prediction_proba_df_0": "prediction_proba_df_loose", 
        "prediction_proba_df_1": "prediction_proba_df_win"})

    opponent_features = ['id', 'tm' ,'opp', 'prediction_proba_df_loose', 'prediction_proba_df_win']

    test_df_w_pred_opp = test_df_w_pred[opponent_features]

    test_df_w_pred = pd.merge(
        test_df_w_pred,
        test_df_w_pred_opp,
        how='left',
        left_on=['id', 'tm' ,'opp'],
        right_on=['id', 'opp', 'tm']
        )

    test_df_w_pred = rename_opponent_columns(test_df_w_pred)

    test_df_w_pred['pred_results_1_line_game'] = np.where(
        test_df_w_pred['prediction_proba_df_win'] > test_df_w_pred['prediction_proba_df_win_opp'], 
        1, 
        0)

    test_df_w_pred = test_df_w_pred.drop_duplicates(subset=['id'], keep='first')

    evaluation_one_line_per_game_precision = round(precision_score(test_df_w_pred['results'], test_df_w_pred['pred_results_1_line_game']),3)
    evaluation_one_line_per_game_accuracy = round(accuracy_score(test_df_w_pred['results'], test_df_w_pred['pred_results_1_line_game']),3)

    logger.info(f"EVALUATION 1 LINE PER GAME PRECISION: {round(evaluation_one_line_per_game_precision, 3)};")
    logger.info(f"EVALUATION 2 LINE PER GAME ACCURACY: {round(evaluation_one_line_per_game_accuracy, 3)};")

    #---------------------------------------------

    evaluation_precision = precision_score(y_test, prediction_value)
    evaluation_accuracy = accuracy_score(y_test, prediction_value)

    logger.info(f"EVALUATION 2LINES PER GAME PRECISION: {round(evaluation_precision, 3)};")
    logger.info(f"EVALUATION 2LINES PER GAME ACCURACY: {round(evaluation_accuracy, 3)};")

    report = {
        "cv_accuracy": round(cv_accuracy, 3),
        "cv_precision": round(cv_precision, 3),
        "evaluation_2lg_accuracy": round(evaluation_accuracy, 3),
        "evaluation_2lg_precision": round(evaluation_precision, 3),
        "evaluation_1lg_accuracy": round(evaluation_one_line_per_game_accuracy, 3),
        "evaluation_1lg_precision": round(evaluation_one_line_per_game_precision, 3),
        "actual": y_test,
        "predicted": prediction_value,
    }

    logger.info("Save metrics")

    with open('data/reports/metrics.json', "w", encoding="utf-8") as fp:
        json.dump(
            obj={
                "accuracy_cv_score": report["cv_accuracy"],
                "precision_cv_score": report["cv_precision"],
                "accuracy_2lg_evaluation_score": report["evaluation_2lg_accuracy"],
                "precision_2lg_evaluation_score": report["evaluation_2lg_precision"],
                "accuracy_1lg_evaluation_score": report["evaluation_1lg_accuracy"],
                "precision_1lg_evaluation_score": report["evaluation_1lg_precision"],
            },
            fp=fp,
        )

    logger.info(f"Accuracy & Precision metrics file saved to : {'data/reports/metrics.json'}")

    # Save shape value plot
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