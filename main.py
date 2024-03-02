import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import yaml
from src import prep, train, inference
import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Prep train data
training_data_path = "./data/train.csv"
output_training_path = "./data/prep.csv"

df_train = prep.load_and_process_training_data(training_data_path, output_training_path)

# Prep inference data
inference_data_path = "./data/test.csv"
common_columns_yaml_path = "./artifacts/common_columns.yaml"
output_inference_path = "./data/inference.csv"

df_inference = prep.load_and_process_inference_data(inference_data_path, common_columns_yaml_path, output_inference_path)

# Trains model
file_path_training = "./data/prep.csv"
output_model_path = "./artifacts/xgboost_model.joblib"
output_params_path = "./artifacts/params.yaml"

best_model = train.load_data_and_train_model(file_path_training, output_model_path, output_params_path)

# Makes inference
input_csv_path = "./data/inference.csv"
model_path = "./artifacts/xgboost_model.joblib"
output_csv_path = "./data/prediction.csv"

output_predictions_df = inference.make_predictions_and_save(input_csv_path, model_path, output_csv_path)
