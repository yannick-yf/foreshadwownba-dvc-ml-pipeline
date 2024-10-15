"""
This module downloads the training dataset from a MySQL database.

References:
    - https://sebrave.medium.com/how-to-spin-up-a-local-mysql-database-on-macos-a550918f092b
    - https://blog.devart.com/delete-duplicate-rows-in-mysql.html
    - https://numberly.tech/orchestrating-python-workflows-in-apache-airflow-fd8be71ad504
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.utils.logs import get_logger


def get_training_dataset(config_path: Path) -> pd.DataFrame:
    """
    Load raw data from the MySQL database.

    Args:
        config_path (Text): Path to the configuration file.

    Returns:
        pd.DataFrame: The training dataset.
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)

    # Loading variables from the .env file
    load_dotenv()

    nba_training_dataset = pd.read_csv("./nba_training_dataset.csv")

    # logger = get_logger(
    #     "DOWNLOAD_TRAINING_DATA_FROM_DATABASE", log_level=config_params["base"]["log_level"]
    # )

    # engine = create_engine(
    #     f"mysql+pymysql://{os.getenv('MYSQL_USERNAME')}:{os.getenv('MYSQL_PASSWORD')}"
    #     f"@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DATABASE')}"
    # )

    # nba_training_dataset = pd.read_sql(
    #     """
    #     SELECT *
    #     FROM foreshadwownba.training_dataset;
    #     """,
    #     engine,
    # )

    training_set = config_params["get_training_dataset"]["training_dataset"]

    nba_training_dataset.to_csv(
        training_set, index=False
    )

    logger.info("Download training dataset from the database is complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    get_training_dataset(config_path=args.config_params)
