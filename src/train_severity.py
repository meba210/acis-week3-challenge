import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main(input_path="data/raw.csv", output_path="models/severity_model.pkl"):
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(input_path)

    X = df.drop(columns=["charges"])
    y = df["charges"]

    numeric = ["age", "bmi", "children"]
    categorical = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    X_pre = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pre, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    print("Model trained.")
    print("RMSE:", rmse)
    print("MAE:", mae)

    joblib.dump((model, preprocessor), output_path)
    print("Severity model saved to", output_path)

if __name__ == "__main__":
    main()
