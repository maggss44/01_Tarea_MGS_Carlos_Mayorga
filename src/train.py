"""
train.py

Este script carga un DataFrame desde un archivo CSV, realiza el split de los datos, entrena 
un modelo XGBoost utilizando RandomizedSearchCV, guarda el modelo entrenado 
en un archivo utilizando joblib y guarda los hiperparámetros en un archivo YAML.

Cómo utilizar:
  python train.py file_path output_model_path output_params_path

Parámetros:
  - file_path (str): Ruta al archivo CSV que contiene los datos de entrenamiento.
  - output_model_path (str): Ruta al archivo donde se guardará el modelo entrenado.
  - output_params_path (str): Ruta al archivo YAML donde se guardarán los hiperparámetros.

Dependencias:
  - pandas
  - xgboost
  - scikit-learn
  - joblib
  - yaml
  - logging

Funciones:
  - load_data_and_train_model(file_path, output_model_path, output_params_path)
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import yaml
import logging

# Configuración de logging
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def load_data_and_train_model(file_path, output_model_path, output_params_path):
    """
    Carga un DataFrame desde un archivo CSV, realiza el split de los datos, entrena 
    un modelo XGBoost utilizando RandomizedSearchCV, guarda el modelo entrenado 
    en un archivo utilizando joblib y guarda los hiperparámetros en un archivo YAML.

    Parameters:
    - file_path (str): Ruta al archivo CSV que contiene los datos de entrenamiento.
    - output_model_path (str): Ruta al archivo donde se guardará el modelo entrenado.
    - output_params_path (str): Ruta al archivo YAML donde se guardarán los hiperparámetros.

    Returns:
    - best_estimator (XGBRegressor): Mejor modelo entrenado.
    """
    try:
        # Lee el DataFrame desde el archivo CSV
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {file_path} no fue encontrado.")
        return None

    # Separa las características y la variable objetivo
    y = df['SalePrice']
    X = df.drop('SalePrice', axis=1)

    # Inicializa el modelo XGBoost
    model = XGBRegressor()

    # Definir el espacio de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 1, 2],
    }

    # Configurar la búsqueda aleatoria
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    # Realizar la búsqueda aleatoria
    random_search.fit(X, y)

    # Obtener el mejor modelo
    best_estimator = random_search.best_estimator_

    # Guardar el mejor modelo en un archivo
    joblib.dump(best_estimator, output_model_path)
    logging.info(f"Modelo entrenado guardado en {output_model_path}")

    # Guardar los hiperparámetros en un archivo YAML
    with open(output_params_path, 'w') as file:
        yaml.dump(random_search.best_params_, file)
    logging.info(f"Hiperparámetros guardados en {output_params_path}")

    return best_estimator
