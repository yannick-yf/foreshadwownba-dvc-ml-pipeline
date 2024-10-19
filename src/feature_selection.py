"""
Feature Selection Step
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from auto_feat_ml import FeatureSelection
from auto_feat_ml.data_models.feature_model import FeatureIn

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

    # -----------------------------------------------
    # Read params for the evaluation steps
    # -----------------------------------------------
    target_column = config_params["dummy_classifier"]["target_variable"]
    group_cv_variable = config_params["data_split"]["group_cv_variable"]
    cross_validation_n_splits = config_params["feature_selection"][
        "cross_validation_n_splits"
    ]
    list_nb_feature_to_select = config_params["feature_selection"][
        "list_nb_feature_to_select"
    ]
    features_to_force = config_params["feature_selection"]["features_to_force"]

    logger.info("Load train dataset")
    train_df = pd.read_csv("./data/processed/train_dataset.csv")
    test_df = pd.read_csv("./data/processed/test_dataset.csv")

    x_train = train_df.drop(
        [target_column, group_cv_variable, "id_season", "tm", "opp"], axis=1
    )

    y_train = train_df[target_column]

    if config_params["feature_selection"]["list_manual_features_to_select"] is not None:

        list_features_to_select = config_params["feature_selection"][
            "list_manual_features_to_select"
        ]
        x_train = x_train[list_features_to_select]

        output_column_names = list_features_to_select

    if config_params["feature_selection"]["method"] == "automatic":

        # ------------------------------------------
        # list_nb_feature_to_select process from str input to list
        list_nb_feature_to_select = list_nb_feature_to_select.replace("/", ",").split(
            ","
        )
        list_nb_feature_to_select = [int(i) for i in list_nb_feature_to_select]

        groups = train_df[group_cv_variable]
        cross_validation_object = GroupKFold(n_splits=cross_validation_n_splits)

        # ------------------------------------------
        # features_to_force process from str input to list
        if features_to_force is not None:
            features_to_force = features_to_force.replace("/", ",").split(",")
            features_to_force = [str(i) for i in features_to_force]

            feature_selection = FeatureSelection(
                FeatureIn(
                    list_number_feature_to_select=list_nb_feature_to_select,
                    training_set=x_train,
                    target_variable=y_train,
                    features_to_force=features_to_force,
                    selection_type="classification",
                )
            )
        else:
            feature_selection = FeatureSelection(
                FeatureIn(
                    list_number_feature_to_select=list_nb_feature_to_select,
                    training_set=x_train,
                    target_variable=y_train,
                    selection_type="classification",
                )
            )

        output = feature_selection.select_features_pipeline(
            pd_column_groups=groups, group_kfold=cross_validation_object
        )

        logger.info("Column Selected are: %s ", str(output.column_names))

        output_column_names = output.column_names

    # ----------------------------------
    # Filter to get only the columns selected

    id_columns = ["id_season", "tm", "opp"]

    train_fs_df = train_df[id_columns + output_column_names].copy()
    train_fs_df[group_cv_variable] = train_df[group_cv_variable].copy()
    train_fs_df[target_column] = train_df[target_column]

    test_fs_df = test_df[id_columns + output_column_names].copy()
    test_fs_df[group_cv_variable] = test_df[group_cv_variable].copy()
    test_fs_df[target_column] = test_df[target_column]

    train_fs_df.to_csv("./data/processed/train_fs_df.csv")
    test_fs_df.to_csv("./data/processed/test_fs_df.csv")

    column_names_selected_df = pd.DataFrame(output_column_names)
    column_names_selected_df.to_csv("./data/processed/columns_selected.csv")

    logger.info("Feature Selection Step Done")

    logger.info("Save trainsets with feature selected and column selected")

    train_fs_df.to_csv("./data/processed/train_dataset_fs.csv", index=False)
    test_fs_df.to_csv("./data/processed/test_dataset_fs.csv", index=False)

    logger.info("Download training dataset from the database is complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    feature_selection(config_path=args.config_params)
