"""Data split"""

import argparse
import pandas as pd
import yaml

from src.utils.logs import get_logger


def data_split(config_path: dict) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("DATA_SPLIT_STEP", log_level=config["base"]["log_level"])

    dataset = pd.read_csv(config["data_process"]["training_dataset_processed"])

    # --------------------------------
    # Params Reading
    group_cv_variable = config["data_split"]["group_cv_variable"]
    random_state = config["base"]["random_state"]
    split_ratio = config["data_split"]["split_ratio"]

    logger.info("Split features into train and test sets")

    uniques_note_id = pd.DataFrame(
        dataset.groupby([group_cv_variable]).size()
    ).reset_index()[[group_cv_variable]]

    validation_id = (
        uniques_note_id[group_cv_variable]
        .sample(frac=split_ratio, random_state=random_state)
        .reset_index(drop=True)
    )

    test_dataset = dataset[dataset[group_cv_variable].isin(validation_id)]
    train_dataset = dataset[~dataset[group_cv_variable].isin(validation_id)]

    logger.info("split_ratio: %s", split_ratio)
    logger.info("num_train_samples: %s", len(train_dataset))
    logger.info("num_test_samples: %s", len(test_dataset))
    logger.info("num_features: %s", len(train_dataset.columns))

    logger.info("Save train and test sets")

    train_dataset.to_csv("./data/processed/train_dataset.csv", index=False)
    test_dataset.to_csv("./data/processed/test_dataset.csv", index=False)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    data_split(config_path=args.config)
