"""
Feature Selection Step
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from src.utils.logs import get_logger


def feature_selection(config_path: Path) -> pd.DataFrame:
    """
    Load raw data from the MySQL database.

    Args:
        config_path (Text): Path to the configuration file.

    Returns:
        pd.DataFrame: The training dataset.
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)


    logger = get_logger(
        "FEATURE_SELECTION_STEP", log_level=config_params["base"]["log_level"]
    )

    logger.info("Load train dataset")
    train_df = pd.read_csv('./data/processed/train_dataset.csv')
    test_df = pd.read_csv('./data/processed/test_dataset.csv')

    #---------------------------------------------
    # Feature Selection

    list_column_to_select = [
        'results',
        'id', 
        'id_season',
        'tm', 
        'opp',
        # 'game_nb', 
        'before_average_W_ratio',
        'before_average_W_ratio_opp',
        'extdom_ext'
        ]

    train_dataset_fs = train_df[list_column_to_select]
    
    test_dataset_fs = test_df[list_column_to_select]

    logger.info("Save train and test sets")

    train_dataset_fs.to_csv('./data/processed/train_dataset_fs.csv',index=False)
    test_dataset_fs.to_csv('./data/processed/test_dataset_fs.csv',index=False)

    logger.info("Download training dataset from the database is complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    feature_selection(config_path=args.config_params)
