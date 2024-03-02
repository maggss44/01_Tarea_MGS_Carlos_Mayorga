"""
inference.py

Este script carga un conjunto de datos desde un archivo CSV, realiza predicciones utilizando
un modelo previamente entrenado y guarda las predicciones en un nuevo archivo CSV.

Cómo utilizar:
  python inference.py input_csv model_path output_csv

Parámetros:
  - input_csv (str): Ruta al archivo CSV de entrada.
  - model_path (str): Ruta al archivo que contiene el modelo entrenado.
  - output_csv (str): Ruta al archivo CSV de salida para las predicciones.

Dependencias:
  - pandas
  - joblib
  - logging

Funciones:
  - make_predictions_and_save(input_csv, model_path, output_csv)
"""

import pandas as pd
import joblib
import logging

# Configuración de logging
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def make_predictions_and_save(input_csv, model_path, output_csv):
    """
    Carga un conjunto de datos desde un archivo CSV, realiza predicciones utilizando
    un modelo previamente entrenado y guarda las predicciones en un nuevo archivo CSV.

    Parameters:
    - input_csv (str): Ruta al archivo CSV de entrada.
    - model_path (str): Ruta al archivo que contiene el modelo entrenado.
    - output_csv (str): Ruta al archivo CSV de salida para las predicciones.

    Returns:
    - output_df (pd.DataFrame): DataFrame que contiene las predicciones.
    """
    try:
        # Cargar el conjunto de datos de entrada desde el archivo CSV
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {input_csv} no fue encontrado.")
        return None

    # Determinar si la columna 'SalePrice' está presente en el conjunto de datos
    X_eval = df.drop('SalePrice', axis=1, errors='ignore')

    try:
        # Cargar el modelo previamente entrenado
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {model_path} no fue encontrado.")
        return None

    # Realizar predicciones utilizando el modelo cargado
    predictions = loaded_model.predict(X_eval)

    # Crear un DataFrame con las predicciones
    output_df = pd.DataFrame({'Predictions': predictions})

    # Guardar el DataFrame de predicciones en un nuevo archivo CSV
    output_df.to_csv(output_csv, index=False)
    logging.info(f"Predicciones guardadas en {output_csv}")

    return output_df
