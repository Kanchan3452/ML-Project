from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "West_Bengal_PMUY_NFHS5_Combined.xlsx"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_excel(DATA)
columns = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]

X = df[columns].astype(float)
X_scaled = StandardScaler().fit_transform(X)

scores = []
for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append((k, score))
    print(f"K = {k}, Silhouette Score = {score:.3f}")

best_k = max(scores, key=lambda x: x[1])[0]
print("\\nBest number of clusters:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
hierarchical = AgglomerativeClustering(n_clusters=best_k)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

print("\\nDistrict Cluster Assignments:")
print(df[["District", "KMeans_Cluster", "Hierarchical_Cluster"]].to_string(index=False))
print("\\nK-Means Silhouette Score:", round(silhouette_score(X_scaled, df["KMeans_Cluster"]), 3))
print("Hierarchical Silhouette Score:", round(silhouette_score(X_scaled, df["Hierarchical_Cluster"]), 3))

df[["District", "KMeans_Cluster", "Hierarchical_Cluster"]].to_csv(OUT / "clustering_results.csv", index=False)

linked = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df["District"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering of West Bengal Districts")
plt.xlabel("Districts")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(OUT / "hierarchical_dendrogram.png", dpi=300)
plt.close()
print("\\nClustering results saved in:", OUT)
