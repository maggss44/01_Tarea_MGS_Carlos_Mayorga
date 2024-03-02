"""
prep.py

Este script realiza el preprocesamiento de datos de entrenamiento e inferencia, calculando características adicionales,
aplicando one-hot encoding y guardando el DataFrame resultante en un archivo CSV.
También almacena las columnas comunes después de la codificación one-hot en un archivo YAML para asegurar que las 
columnas son iguales en el set de entrenamiento y el set de inferencia.

Cómo utilizar:
  python prep.py training_data output_path
  python prep.py inference_data common_columns_yaml output_path

Parámetros:
  - training_data (str): Ruta al archivo CSV de entrenamiento.
  - output_path (str): Ruta al archivo CSV de salida para el DataFrame preprocesado.
  - inference_data (str): Ruta al archivo CSV de inferencia.
  - common_columns_yaml (str): Ruta al archivo YAML que contiene las columnas comunes.
  - output_path (str): Ruta al archivo CSV de salida para el DataFrame preprocesado de inferencia.

Dependencias:
  - pandas
  - yaml
  - logging

Funciones:
  - calculate_additional_features(df)
  - load_and_process_training_data(training_data, output_path)
  - load_and_process_inference_data(inference_data, common_columns_yaml, output_path)
"""

import pandas as pd
import yaml
import logging

# Configuración de logging
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def calculate_additional_features(df):
    """
    Calcula varias características adicionales y las agrega al DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame a enriquecer con características adicionales.

    Returns:
    - df (pd.DataFrame): DataFrame con características adicionales calculadas.
    """
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
    df['Interaction_OvQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']

    exter_qual_columns = [col for col in df.columns if 'ExterQual' in col]
    bsmt_qual_columns = [col for col in df.columns if 'BsmtQual' in col]
    df['QualityIndex'] = df[exter_qual_columns + bsmt_qual_columns].sum(axis=1)

    df['TotalPorchArea'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

    return df

def load_and_process_training_data(training_data, output_path):
    """
    Carga un DataFrame desde un archivo CSV de entrenamiento, realiza el preprocesamiento,
    guarda el DataFrame resultante en un nuevo archivo CSV y almacena las columnas comunes en un archivo YAML.

    Parameters:
    - training_data (str): Ruta al archivo CSV de entrenamiento.
    - output_path (str): Ruta al archivo CSV de salida para el DataFrame preprocesado.

    Returns:
    - df_raw (pd.DataFrame): DataFrame preprocesado.
    - common_columns (pd.Index): Columnas comunes después de la codificación one-hot.
    """
    try:
        # Lee el DataFrame desde el archivo CSV
        df_raw = pd.read_csv(training_data)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {training_data} no fue encontrado.")
        return None, None

    # Extrae la columna 'SalePrice' como el objetivo y la elimina del DataFrame principal
    y = df_raw['SalePrice']
    df_raw = df_raw.drop('SalePrice', axis=1)

    # Aplica one-hot encoding a las variables categóricas
    df_raw = pd.get_dummies(df_raw, drop_first=True)

    # Guarda las columnas después de la codificación
    common_columns = df_raw.columns
    # Convertir las columnas comunes a una lista
    common_columns_list = common_columns.to_list()

    # Guardar en un archivo YAML
    yaml_path = './artifacts/common_columns.yaml'
    with open(yaml_path, 'w') as file:
        yaml.dump(common_columns_list, file)
    logging.info(f"Columnas comunes guardadas en {yaml_path}")

    # Calcula características adicionales
    df_raw = calculate_additional_features(df_raw)

    # Restaura la columna 'SalePrice' al DataFrame
    df_raw['SalePrice'] = y

    # Guarda el DataFrame preprocesado en un nuevo archivo CSV
    df_raw.to_csv(output_path, index=False)
    logging.info(f"DataFrame preprocesado guardado en {output_path}")

    return df_raw, common_columns

def load_and_process_inference_data(inference_data, common_columns_yaml, output_path):
    """
    Carga un DataFrame desde un archivo CSV de inferencia, realiza el preprocesamiento,
    y guarda el DataFrame resultante en un nuevo archivo CSV.

    Parameters:
    - inference_data (str): Ruta al archivo CSV de inferencia.
    - common_columns_yaml (str): Ruta al archivo YAML que contiene las columnas comunes.
    - output_path (str): Ruta al archivo CSV de salida para el DataFrame preprocesado.

    Returns:
    - df_test (pd.DataFrame): DataFrame preprocesado.
    """
    try:
        # Lee el DataFrame desde el archivo CSV
        df_test = pd.read_csv(inference_data)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {inference_data} no fue encontrado.")
        return None

    # Cargar las columnas comunes desde el archivo YAML
    try:
        with open(common_columns_yaml, 'r') as file:
            common_columns = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: El archivo {common_columns_yaml} no fue encontrado.")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error al cargar el archivo YAML: {e}")
        return None

    # Aplica one-hot encoding al conjunto de prueba
    df_test = pd.get_dummies(df_test, drop_first=True)

    # Asegúrate de que las columnas de df_test sean las mismas que las de df_raw después de one-hot encoding
    df_test = df_test.reindex(columns=common_columns, fill_value=0)

    # Elimina la columna 'SalePrice' si existe
    if 'SalePrice' in df_test.columns:
        df_test = df_test.drop('SalePrice', axis=1)

    # Calcula características adicionales
    df_test = calculate_additional_features(df_test)

    # Guarda el DataFrame preprocesado en un nuevo archivo CSV
    df_test.to_csv(output_path, index=False)
    logging.info(f"DataFrame preprocesado de inferencia guardado en {output_path}")

    return df_test
