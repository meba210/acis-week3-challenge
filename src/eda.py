import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for figures
os.makedirs("figs", exist_ok=True)

# Load Data
df = pd.read_csv("data/insurance.csv")

# =========================
# BASIC DATA SUMMARY
# =========================
print("\nDATA SHAPE:")
print(df.shape)

print("\nMISSING VALUES (%):")
print((df.isna().sum() / len(df)) * 100)

print("\nNUMERICAL SUMMARY:")
print(df.describe())

# =========================
# PLOT 1: AVERAGE CHARGES BY REGION
# =========================
if "region" in df.columns and "charges" in df.columns:
    charges_region = df.groupby("region")["charges"].mean().sort_values(ascending=False)
    charges_region.plot(kind="bar", title="Average Charges by Region", color="skyblue")
    plt.ylabel("Average Charges")
    plt.tight_layout()
    plt.savefig("figs/charges_by_region.png", dpi=150)
    plt.close()

# =========================
# PLOT 2: AVERAGE CHARGES BY SMOKER STATUS
# =========================
if "smoker" in df.columns and "charges" in df.columns:
    charges_smoker = df.groupby("smoker")["charges"].mean()
    charges_smoker.plot(kind="bar", title="Average Charges by Smoker Status", color="salmon")
    plt.ylabel("Average Charges")
    plt.tight_layout()
    plt.savefig("figs/charges_by_smoker.png", dpi=150)
    plt.close()

# =========================
# PLOT 3: AGE VS CHARGES
# =========================
if "age" in df.columns and "charges" in df.columns:
    plt.figure(figsize=(8,5))
    plt.scatter(df["age"], df["charges"], alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.title("Age vs Charges")
    plt.tight_layout()
    plt.savefig("figs/age_vs_charges.png", dpi=150)
    plt.close()

# =========================
# PLOT 4: BMI VS CHARGES
# =========================
if "bmi" in df.columns and "charges" in df.columns:
    plt.figure(figsize=(8,5))
    plt.scatter(df["bmi"], df["charges"], alpha=0.5, color="green")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.title("BMI vs Charges")
    plt.tight_layout()
    plt.savefig("figs/bmi_vs_charges.png", dpi=150)
    plt.close()

# =========================
# PLOT 5: CHILDREN DISTRIBUTION
# =========================
if "children" in df.columns:
    df["children"].value_counts().sort_index().plot(kind="bar", title="Number of Children Distribution", color="purple")
    plt.xlabel("Number of Children")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figs/children_distribution.png", dpi=150)
    plt.close()

print("\nEDA COMPLETE. FIGURES SAVED TO figs/")
