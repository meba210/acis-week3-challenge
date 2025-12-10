import pandas as pd
from scipy.stats import ttest_ind, f_oneway, kruskal
import os

def main(input_path="data/raw.csv", output_path="outputs/hypothesis_results.txt"):
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(input_path)

    lines = []
    lines.append("HYPOTHESIS TEST RESULTS\n")
    lines.append("------------------------\n")

    # Test 1: charges differ by smoker
    smokers = df[df["smoker"] == "yes"]["charges"]
    nonsmokers = df[df["smoker"] == "no"]["charges"]

    t_stat, p_value = ttest_ind(smokers, nonsmokers, equal_var=False)

    lines.append(f"Charges difference between smokers and non-smokers: p-value={p_value}\n")

    # Test 2: charges differ by region
    groups = [grp["charges"].values for _, grp in df.groupby("region")]
    stat, p = f_oneway(*groups)
    lines.append(f"Charges differ by region (ANOVA): p-value={p}\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

    print("Hypothesis testing complete. Results saved to", output_path)

if __name__ == "__main__":
    main()
