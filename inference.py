import pandas as pd
import joblib

df = pd.read_csv("./data/inference.csv")

if 'SalePrice' in df.columns:
    X_eval = df.drop('SalePrice', axis=1)
else:
    X_eval = df

loaded_model = joblib.load("./artifacts/xgboost_model.joblib")

predictions = loaded_model.predict(X_eval)

output_df = pd.DataFrame({'Predictions': predictions})

output_df.to_csv("./data/prediction.csv", index=False)