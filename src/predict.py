import pandas as pd
import joblib
import os

def main(input_path="data/raw.csv", model_path="models/severity_model.pkl", output_path="outputs/predictions.csv"):
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(input_path)
    model, preprocessor = joblib.load(model_path)

    X = df.drop(columns=["charges"])
    X_pre = preprocessor.transform(X)

    preds = model.predict(X_pre)
    df["PredictedCharges"] = preds

    df[["PredictedCharges"]].to_csv(output_path, index=False)
    print("Predictions saved to", output_path)

if __name__ == "__main__":
    main()
