import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def main(model_path="models/severity_model.pkl", input_path="data/raw.csv", output_path="figs/shap_summary.png"):
    os.makedirs("figs", exist_ok=True)

    df = pd.read_csv(input_path)
    model, preprocessor = joblib.load(model_path)

    X = df.drop(columns=["charges"])
    X_pre = preprocessor.transform(X)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_pre[:300])  # sample for speed

    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("SHAP summary saved to", output_path)

if __name__ == "__main__":
    main()
