from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "West_Bengal_PMUY_NFHS5_Combined.xlsx"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_excel(DATA)
print("\\nDATASET SUMMARY")
print("Districts:", df["District"].nunique())
print("Complete years used:", "2016-17 to 2021-22")
print("Partial year excluded from regression:", "2022-23_Apr-Feb")
print("NFHS-5 values available:", df["NFHS5_Clean_Fuel_Percent"].notna().sum())
print("NFHS-5 values missing:", df["NFHS5_Clean_Fuel_Percent"].isna().sum())
print("2020-21 recorded zeros:", int((df["2020-21"] == 0).sum()))

overall_path = OUT / "pooled_chronological_overall_metrics.csv"
if overall_path.exists():
    print("\\nPOOLED CHRONOLOGICAL REGRESSION RESULTS")
    print(pd.read_csv(overall_path).round(4).to_string(index=False))
else:
    print("\\nRun 07_model_comparison.py first to generate regression metrics.")

cluster_path = OUT / "clustering_results.csv"
if cluster_path.exists():
    print("\\nCLUSTERING RESULTS")
    c = pd.read_csv(cluster_path)
    print("K-Means districts per cluster:")
    print(c.groupby("KMeans_Cluster")["District"].apply(list).to_string())
else:
    print("\\nRun 05_clustering.py before reviewing cluster assignments.")

nfhs = df.dropna(subset=["NFHS5_Clean_Fuel_Percent"])
if not nfhs.empty:
    print("\\nNFHS-5 CLEAN-FUEL SUMMARY")
    print("Mean:", round(nfhs["NFHS5_Clean_Fuel_Percent"].mean(), 2), "%")
    print("Highest:", nfhs.loc[nfhs["NFHS5_Clean_Fuel_Percent"].idxmax(), ["District", "NFHS5_Clean_Fuel_Percent"]].to_dict())
    print("Lowest:", nfhs.loc[nfhs["NFHS5_Clean_Fuel_Percent"].idxmin(), ["District", "NFHS5_Clean_Fuel_Percent"]].to_dict())
