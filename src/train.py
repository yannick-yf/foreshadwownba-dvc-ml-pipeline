
import pandas as pd
from typing import Text
import yaml
import argparse
from decouple import config
import os
import pprint

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.training.train_cross_val import train_cross_val
from src.utils.logs import get_logger
import json
from sklearn.preprocessing import OneHotEncoder

def train(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
        

    # -----------------------------------------------
    # Read params for the training steps
    # -----------------------------------------------
    target_column = config['dummy_classifier']['target_variable']
    group_cv_variable = config['data_split']['group_cv_variable']
    estimator_configs = config['train']['param_grid']
    cross_validation_n_splits = config['train']['cross_validation_n_splits']
    scoring_metric = config['train']['scoring_metric']
    model_name = config['train']['model_name']
    
    logger = get_logger(
        'TRAINING_STEP', 
        log_level=config['base']['log_level'])

    logger.info("Get estimator name")
    logger.info(f"Estimator: {model_name}")

    # if model_name:
    # # Filter the estimator_configs to include only the top 3 models
    #     filtered_estimator_configs = [
    #         config for config in estimator_configs if config["name"] == model_name
    #     ]
    #     if not filtered_estimator_configs:
    #         logger.error(f"Invalid model name: {model_name}")
    #         return
    # else:
    #     top_3_models_df = pd.read_csv(top_3_models)
    #     top_3_models = top_3_models_df["Model"].tolist()
    #     filtered_estimator_configs = [
    #         config for config in estimator_configs if config["name"] in top_3_models
    #     ] 

    # logger.info(f"Filtered estimator configs: {filtered_estimator_configs}")


    logger.info("Load train dataset")
    train_df = pd.read_csv('./data/processed/train_dataset.csv')
    groups = train_df[group_cv_variable]

    # Split the training data into features (X_train) and target (y_train)
    train_df = train_df.reset_index(drop=True)
    logger.info(f"cross_validation_n_splits: {cross_validation_n_splits}")

    logger.info("Train Cross-Validation to get best params")
    scores = train_cross_val(
        train_df=train_df,
        target_column=target_column,
        group_cv_variable=group_cv_variable,
        estimator_configs=estimator_configs,# filtered_estimator_configs,
        cross_validation_n_splits=cross_validation_n_splits,
        scoring_metric=scoring_metric,
        groups=groups
    )

    # # Fit the model on the training data
    # logger.info("Fitting the model on the training data...")
    # best_estimator.fit(X_train, y_train)
    # logger.info("Model fitted successfully.")

    # # Print the number of features in X_train
    # print(f"Number of features in X_train: {X_train.shape[1]}")

    # # Print the expected number of features for the best estimator
    # print(f"Expected number of features for {best_estimator}: {best_estimator.n_features_in_}")

    # Create a new dictionary without the 'estimator' key
    # best_scores_modified = {k: v for k, v in scores.items() if k != 'estimator'}

    # logger.info("Best scores:")
    # pprint.pprint(best_scores_modified, indent=2)


    # Create the DataFrame
    cross_val_scores_df = pd.DataFrame(scores)
    
    cross_val_scores_df.to_csv('./models/cross_val_score.csv',index=False)

    logger.info("Save best model")

    # try:
    #     joblib.dump(best_estimator, './models/model.joblib')
    #     logger.info("Best model saved successfully.")
    # except Exception as e:
    #     logger.error(f"Error saving model: {e}")


    # # Load the trained model
    # model = joblib.load('models/model.joblib')
    # #logger.info(f"Loaded model type: {type(model)}")
    # try:
    #     model = joblib.load('models/model.joblib')
    #     logger.info(f"Loaded model type: {type(model)}")
    # except Exception as e:
    #     logger.error(f"Error loading model: {e}")

    cv_accuracy = round(cross_val_scores_df.test_accuracy_score.mean(),3)
    cv_precision = round(cross_val_scores_df.test_precision_score.mean(),3)

    logger.info(f"CV ACCURACY: {cv_accuracy};")
    logger.info(f"CV PRECISION: {cv_precision};")

    logger.info("Train Step Complete")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--config', 
        dest='config', 
        required=True)

    
    args = arg_parser.parse_args()

    train(
        config_path = args.config,
        )
