import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def main(predictions_path="outputs/predictions.csv", truth_path="data/raw.csv", output_path="outputs/model_metrics.txt"):
    os.makedirs("outputs", exist_ok=True)

    preds = pd.read_csv(predictions_path)
    truth = pd.read_csv(truth_path)

    rmse = mean_squared_error(truth["charges"], preds["PredictedCharges"], squared=False)
    mae = mean_absolute_error(truth["charges"], preds["PredictedCharges"])

    lines = [
        f"RMSE: {rmse}\n",
        f"MAE: {mae}\n"
    ]

    with open(output_path, "w") as f:
        f.writelines(lines)

    print("Evaluation metrics saved to", output_path)

if __name__ == "__main__":
    main()
