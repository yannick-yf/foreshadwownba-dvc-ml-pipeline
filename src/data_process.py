"""Data Process"""

import argparse
from typing import Text

import pandas as pd
import yaml

from src.utils.logs import get_logger

def data_process(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("DATA_SPLIT_STEP", log_level=config["base"]["log_level"])

    dataset = pd.read_csv(config["get_training_dataset"]["training_dataset"]).head(1000)

    # Manual feature selection and delation
    list_column_to_drop = config["data_process"]["list_column_to_drop"]
    dataset_processed = dataset.drop(list_column_to_drop, axis=1)

    dataset_processed.to_csv(
        config["data_process"]["training_dataset_processed"], index=False
    )

    logger.info("Training dataset processed and saved")



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    data_process(config_path=args.config)
