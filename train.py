import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib


df = pd.read_csv("./data/prep.csv")

X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']

model = XGBRegressor()

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 2],
}

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X, y)

joblib.dump(random_search.best_estimator_, "./artifacts/xgboost_model.joblib")