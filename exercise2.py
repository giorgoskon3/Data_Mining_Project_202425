import dask.dataframe as dd
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def separator(title: str = None):
    if title:
        print(60*"-")
        print(title)
        print(60*"-")
    else:
        print(60*"-")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        ddf = dd.read_csv(file_path, blocksize="16MB", dtype="object", assume_missing=True)
        df_sample = df_sample.convert_dtypes()
        separator("Data Loaded")
        return df_sample
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def has_missing_values(df: dd.DataFrame) -> bool:
    total_missing = df.isnull().sum().sum().compute()
    print(f"Συνολικές ελλείψεις (NaN): {total_missing}")
    return total_missing > 0

# Preprocessing 
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    separator("Preprocessing")

    # Βήμα 1: Αφαίρεση σταθερών στηλών
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"Αφαίρεση σταθερών στηλών: {constant_cols}")
    df = df.drop(columns=constant_cols, errors="ignore")

    # Βήμα 2: Προσπάθησε να μετατρέψεις όλες τις στήλες σε αριθμητικές όπου γίνεται
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # όσες δεν μετατρέπονται θα γίνουν NaN

    # Βήμα 3: Κράτα μόνο τις στήλες που είναι πλέον αριθμητικές
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("Δεν βρέθηκαν αριθμητικά γνωρίσματα για επεξεργασία.")
        return pd.DataFrame()

    # Βήμα 4: Αφαίρεση πολύ συσχετισμένων χαρακτηριστικών (>0.9)
    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    if to_drop:
        print(f"Αφαίρεση λόγω υψηλής συσχέτισης: {to_drop}")
    df_numeric = df_numeric.drop(columns=to_drop, errors="ignore")
    
    df_numeric = df_numeric.dropna(axis=1, how='all')  # Αφαίρεση στηλών με όλα τα NaN

    return df_numeric

def create_sample(df: dd.DataFrame, frac=0.1, label_col=None):
    separator("Creating Sample")
    SAMPLE_OUTPUT = "sample.csv"
    if label_col and label_col in df.columns:
        sample = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(frac=frac))
    else:
        sample = df.sample(frac=frac)
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
    df_numeric = df.select_dtypes(include="number")
    if df_numeric.empty:
        print("Δεν υπάρχουν αριθμητικά δεδομένα για clustering.")
        return None

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
    
    df = preprocessing(df)
    
    create_sample(df, frac=0.1)
    
    best_k = find_best_k_silhouette(df, k_range=range(9, 15))
    create_kmeans(df, n_clusters=best_k)
    create_agglo(df, n_clusters=best_k)
