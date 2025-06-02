import dask.dataframe as dd
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def separator(title: str = None):
    if title:
        print(60*"-")
        print(title)
        print(60*"-")
    else:
        print(60*"-")

def load_data(file_path: str) -> dd.DataFrame:
    try:
        df = dd.read_csv(file_path, blocksize="16MB", dtype="object", assume_missing=True)
        separator("Data Loaded")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def has_missing_values(df: dd.DataFrame) -> bool:
    total_missing = df.isnull().sum().sum().compute()
    print(f"Συνολικές ελλείψεις (NaN): {total_missing}")
    return total_missing > 0

# Preprocessing
def preprocessing(df: dd.DataFrame) -> None:
    separator("Preprocessing")
    
    # Drop columns with constant values
    nunique = df.nunique().compute()
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols, errors="ignore")
    
    # Drop columns with high correlation (>0.9)
    df_sample = df.sample(frac=0.01).compute()
    df_sample = df_sample.convert_dtypes()
    numeric = df_sample.select_dtypes(include=["number"]).dropna()
    corr = numeric.corr()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                to_drop.add(corr.columns[j])
    df = df.drop(columns=list(to_drop), errors="ignore")
    cols_to_remove = ["Label", "Traffic Type", "Traffic Subtype"]
    df = df.drop(columns=cols_to_remove, errors="ignore")

def create_sample(df: dd.DataFrame, frac=0.1, label_col=None):
    separator("Creating Sample")
    SAMPLE_OUTPUT = "sample.csv"
    if label_col and label_col in df.columns:
        sample = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(frac=frac)).compute()
    else:
        sample = df.sample(frac=frac).compute()
    sample.to_csv(SAMPLE_OUTPUT, index=False)
    print(f"Sample saved to {SAMPLE_OUTPUT}")
    return sample

def create_kmeans(df: pd.DataFrame, n_clusters=10):
    separator("KMeans Clustering")
    KMEANS_OUTPUT = "kmeans.csv"
    df_numeric = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_numeric["Cluster"] = kmeans.fit_predict(X_scaled)

    centroids = []
    for i in range(n_clusters):
        cluster_i = df_numeric[df_numeric["Cluster"] == i].drop("Cluster", axis=1)
        center = kmeans.cluster_centers_[i]
        closest_idx = ((cluster_i - center) ** 2).sum(axis=1).idxmin()
        centroids.append(cluster_i.loc[closest_idx])

    kmeans_df = pd.DataFrame(centroids)
    kmeans_df.to_csv(KMEANS_OUTPUT, index=False)
    print(f"KMeans saved to {KMEANS_OUTPUT}")
    return kmeans_df

# Finding the best K using Silhouette Score
def find_best_k_silhouette(df: pd.DataFrame, k_range=range(2, 15)):
    separator("Finding Best K using Silhouette Score")
    df_numeric = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"K = {k}: Silhouette Score = {score:.4f}")

    best_k = k_range[scores.index(max(scores))]
    print(f"\nΒέλτιστο K με βάση το Silhouette Score: {best_k}")

    # Προαιρετικά: γράφημα
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), scores, marker='o')
    plt.xlabel("Αριθμός Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score για διαφορετικά K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("silhouette_plot.png", dpi=300)
    plt.show()

    return best_k

def create_agglo(df: pd.DataFrame, n_clusters=10):
    separator("Agglomerative Clustering")
    AGGLO_OUTPUT = "agglo.csv"
    df_numeric = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    df_numeric["Cluster"] = agglo.fit_predict(X_scaled)

    centroids = []
    for i in range(n_clusters):
        cluster_i = df_numeric[df_numeric["Cluster"] == i].drop("Cluster", axis=1)
        center = cluster_i.mean()
        closest_idx = ((cluster_i - center) ** 2).sum(axis=1).idxmin()
        centroids.append(cluster_i.loc[closest_idx])

    agglo_df = pd.DataFrame(centroids)
    agglo_df.to_csv(AGGLO_OUTPUT, index=False)
    print(f"Agglomerative saved to {AGGLO_OUTPUT}")
    return agglo_df

if __name__ == "__main__":
    df = load_data("data.csv")
    
    # Uncomment to check for missing values
    # has_missing_values(df)
    
    # preprocessing(df)
    