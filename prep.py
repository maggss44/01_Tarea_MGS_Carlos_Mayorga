""" import pandas as pd

# Lee el DataFrame desde el archivo CSV
df_raw = pd.read_csv("./data/raw.csv")

y = df_raw['SalePrice']
df_raw = df_raw.drop('SalePrice', axis = 1)

# Aplica one-hot encoding a las variables categóricas
df_raw = pd.get_dummies(df_raw, drop_first=True)

# Calcula la edad de la casa, el área total y el número total de baños
df_raw['HouseAge'] = df_raw['YrSold'] - df_raw['YearBuilt']
df_raw['TotalSF'] = df_raw['TotalBsmtSF'] + df_raw['1stFlrSF'] + df_raw['2ndFlrSF']
df_raw['TotalBathrooms'] = df_raw['FullBath'] + 0.5 * df_raw['HalfBath']

# Calcula la interacción entre OverallQual y GrLivArea
df_raw['Interaction_OvQual_GrLivArea'] = df_raw['OverallQual'] * df_raw['GrLivArea']

# Filtra las columnas relacionadas con ExterQual y BsmtQual
exter_qual_columns = [col for col in df_raw.columns if 'ExterQual' in col]
bsmt_qual_columns = [col for col in df_raw.columns if 'BsmtQual' in col]

# Calcula el índice de calidad
df_raw['QualityIndex'] = df_raw[exter_qual_columns + bsmt_qual_columns].sum(axis=1)

# Calcula el área total de los porches
df_raw['TotalPorchArea'] = df_raw['OpenPorchSF'] + df_raw['EnclosedPorch'] + df_raw['3SsnPorch'] + df_raw['ScreenPorch']

# Guarda el DataFrame preprocesado en un nuevo archivo CSV
df_raw.to_csv('./data/prep.csv', index=False)


# Lee el DataFrame desde el archivo CSV
df_test = pd.read_csv("./data/test.csv")

# Aplica one-hot encoding al conjunto de prueba usando solo las columnas presentes en el conjunto de entrenamiento
df_test = pd.get_dummies(df_test, drop_first=True)
df_test = df_test[df_raw.columns]

# Calcula la edad de la casa, el área total y el número total de baños
df_test['HouseAge'] = df_test['YrSold'] - df_test['YearBuilt']
df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
df_test['TotalBathrooms'] = df_test['FullBath'] + 0.5 * df_test['HalfBath']

# Calcula la interacción entre OverallQual y GrLivArea
df_test['Interaction_OvQual_GrLivArea'] = df_test['OverallQual'] * df_test['GrLivArea']

# Filtra las columnas relacionadas con ExterQual y BsmtQual
exter_qual_columns = [col for col in df_test.columns if 'ExterQual' in col]
bsmt_qual_columns = [col for col in df_test.columns if 'BsmtQual' in col]

# Calcula el índice de calidad
df_test['QualityIndex'] = df_test[exter_qual_columns + bsmt_qual_columns].sum(axis=1)

# Calcula el área total de los porches
df_test['TotalPorchArea'] = df_test['OpenPorchSF'] + df_test['EnclosedPorch'] + df_test['3SsnPorch'] + df_test['ScreenPorch']

# Guarda el DataFrame preprocesado en un nuevo archivo CSV
df_test.to_csv('./data/inference.csv', index=False) """

import pandas as pd

# Lee el DataFrame desde el archivo CSV
df_raw = pd.read_csv("./data/raw.csv")

y = df_raw['SalePrice']
df_raw = df_raw.drop('SalePrice', axis=1)

# Aplica one-hot encoding a las variables categóricas
df_raw = pd.get_dummies(df_raw, drop_first=True)

# Guarda las columnas después de la codificación
common_columns = df_raw.columns

# Calcula la edad de la casa, el área total y el número total de baños
df_raw['HouseAge'] = df_raw['YrSold'] - df_raw['YearBuilt']
df_raw['TotalSF'] = df_raw['TotalBsmtSF'] + df_raw['1stFlrSF'] + df_raw['2ndFlrSF']
df_raw['TotalBathrooms'] = df_raw['FullBath'] + 0.5 * df_raw['HalfBath']

# Calcula la interacción entre OverallQual y GrLivArea
df_raw['Interaction_OvQual_GrLivArea'] = df_raw['OverallQual'] * df_raw['GrLivArea']

# Filtra las columnas relacionadas con ExterQual y BsmtQual
exter_qual_columns = [col for col in df_raw.columns if 'ExterQual' in col]
bsmt_qual_columns = [col for col in df_raw.columns if 'BsmtQual' in col]

# Calcula el índice de calidad
df_raw['QualityIndex'] = df_raw[exter_qual_columns + bsmt_qual_columns].sum(axis=1)

# Calcula el área total de los porches
df_raw['TotalPorchArea'] = df_raw['OpenPorchSF'] + df_raw['EnclosedPorch'] + df_raw['3SsnPorch'] + df_raw['ScreenPorch']

df_raw['SalePrice'] = y

# Guarda el DataFrame preprocesado en un nuevo archivo CSV
df_raw.to_csv('./data/prep.csv', index=False)


# Lee el DataFrame desde el archivo CSV
df_test = pd.read_csv("./data/test.csv")

# Aplica one-hot encoding al conjunto de prueba
df_test = pd.get_dummies(df_test, drop_first=True)

# Asegúrate de que las columnas de df_test sean las mismas que las de df_raw después de one-hot encoding
df_test = df_test.reindex(columns=common_columns, fill_value=0)

# Calcula la edad de la casa, el área total y el número total de baños
df_test['HouseAge'] = df_test['YrSold'] - df_test['YearBuilt']
df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
df_test['TotalBathrooms'] = df_test['FullBath'] + 0.5 * df_test['HalfBath']

# Calcula la interacción entre OverallQual y GrLivArea
df_test['Interaction_OvQual_GrLivArea'] = df_test['OverallQual'] * df_test['GrLivArea']

# Filtra las columnas relacionadas con ExterQual y BsmtQual
exter_qual_columns = [col for col in df_test.columns if 'ExterQual' in col]
bsmt_qual_columns = [col for col in df_test.columns if 'BsmtQual' in col]

# Calcula el índice de calidad
df_test['QualityIndex'] = df_test[exter_qual_columns + bsmt_qual_columns].sum(axis=1)

# Calcula el área total de los porches
df_test['TotalPorchArea'] = df_test['OpenPorchSF'] + df_test['EnclosedPorch'] + df_test['3SsnPorch'] + df_test['ScreenPorch']

# Guarda el DataFrame preprocesado en un nuevo archivo CSV
df_test.to_csv('./data/inference.csv', index=False)
