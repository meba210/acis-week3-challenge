# src/train_frequency.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

def find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None

def prepare_features(df, id_col=None, drop_cols=None):
    df = df.copy()
    if drop_cols is None:
        drop_cols = []
    # simple feature engineering: numeric only, fillna with median
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in drop_cols]
    X = df[numeric].fillna(df[numeric].median())
    return X

def main(input_path, model_out):
    df = pd.read_csv(input_path)
    claims_col = find_col(df, ["claim", "claims", "totalclaim", "claimamount"])
    if claims_col is None:
        raise KeyError("Claims column not found.")
    df["HasClaim"] = (df[claims_col] > 0).astype(int)

    # Drop identifier-like non predictors
    id_col = find_col(df, ["id", "policy", "policyid"])
    drop_cols = [claims_col]
    if id_col:
        drop_cols.append(id_col)

    X = prepare_features(df, id_col=id_col, drop_cols=drop_cols)
    y = df["HasClaim"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    # evaluation
    y_prob = model.predict_proba(X_val)[:,1]
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_prob)

    print(f"Freq model: AUC={auc:.4f}, Acc={acc:.4f}, LogLoss={ll:.4f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(model, model_out)
    print("Frequency model saved to", model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw.csv")
    parser.add_argument("--model-out", default="models/frequency_model.pkl")
    args = parser.parse_args()
    main(args.input, args.model_out)
