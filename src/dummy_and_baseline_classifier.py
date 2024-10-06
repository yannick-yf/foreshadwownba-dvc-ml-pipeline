"""Dummy Classifier"""

import os
from pathlib import Path
import pandas as pd
import argparse
from typing import Text
import yaml
from decouple import config
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sys

from src.utils.logs import get_logger

def dummy_and_baseline_classifier(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger(
        'DUMMY_CLASSIFIER_STEP', 
        log_level=config['base']['log_level'])
        
    #-----------------------------------------------
    # Read input params
    target_column = config['dummy_classifier']['target_variable']
    random_state = config['base']['random_state']

    logger.info("Load train dataset")
    train_df = pd.read_csv('./data/processed/train_dataset.csv')
    test_df = pd.read_csv('./data/processed/test_dataset.csv')


    logger.info("Multiple Models Pre Train:")

    X_train = train_df.drop([target_column], axis=1)
    y_train = train_df[target_column]
    
    #--------------------------------
    # Dummy Classifier

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    DummyClassifier(strategy='most_frequent')
    dummy_clf.predict(X_train)
    dummy_classifier_score = round(dummy_clf.score(X_train, y_train), 3)

    logger.info(f"Dummy Classifier Score: {dummy_classifier_score}")

    #--------------------------------
    # Baseline Classifier - Simple Domain expert rules

    test_df_processed = pd.merge(
        test_df[['id', 'before_average_W_ratio', 'tm', 'opp', 'results']],
        test_df[['id', 'before_average_W_ratio', 'tm', 'opp']],
        how='left',
        on='id')

    # Remove duplciate - comditon is where tm_x != tm_y
    test_df_processed = test_df_processed[test_df_processed['tm_x'] != test_df_processed['tm_y']]

    test_df_processed['benchmark_prob'] = np.where(
        test_df_processed['before_average_W_ratio_x'] > test_df_processed['before_average_W_ratio_y'], 
        1, 
        0)

    tn, fp, fn, tp = confusion_matrix(test_df_processed['results'], test_df_processed['benchmark_prob']).ravel()
    precision_metric = round(tp / (tp + fp), 3)
    specificity_metric = round(tn / (tn+fp), 3)
    recall_metric = round(tp / (tp + fn), 3)
    f1_metric = round((2 * precision_metric * recall_metric) / (precision_metric + recall_metric), 3)
    accuracy_metric = round((tp + tn) / (tp + tn + fp + fn), 3)

    logger.info(f"Baseline Accuracy: {accuracy_metric}")
    logger.info(f"Baseline Precision: {precision_metric}")
    logger.info(f"Baseline Specificity: {specificity_metric}")
    logger.info(f"Baseline Recall: {recall_metric}")
    logger.info(f"Baseline F1: {f1_metric}")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--config', 
        dest='config', 
        required=True)
    
    args = arg_parser.parse_args()

    dummy_and_baseline_classifier(
        config_path = args.config
        )
