import dask.dataframe as dd
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import hdbscan

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
def preprocessing(df: dd.DataFrame) -> pd.DataFrame:
    separator("Preprocessing")

    # Remove Constant Columns
    nunique = df.nunique().compute()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"Remove Constant Columns: {constant_cols}")
    df = df.drop(columns=constant_cols)

    # Convert to pandas DataFrame
    df = df.compute()

    # Save Categorical data
    known_categoricals = [col for col in df.columns if col.lower() in ["label", "traffic type", "traffic subtype"]]
    print(f"Saved Categorical Columns: {known_categoricals}")
    preserved_categoricals = df[known_categoricals].copy()

    # Convert to numeric where possible
    for col in df.columns:
        if col not in known_categoricals:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Choose numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = upper.stack().sort_values(ascending=False)
    print("\nHighly Correlated Columns > 0.9:")
    print(high_corr_pairs[high_corr_pairs > 0.9])

    # Remove highly correlated columns (>0.9)
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    if to_drop:
        print(f"\nRemoved highly correlated columns: {to_drop}")
        numeric_df = numeric_df.drop(columns=to_drop)

    # Concatenate numeric and preserved categorical data
    final_df = pd.concat([numeric_df.reset_index(drop=True), preserved_categoricals.reset_index(drop=True)], axis=1)
    final_df = final_df.dropna(axis=1)
    
    # Save the preprocessed DataFrame
    final_df.to_csv("preprocessed_data.csv", index=False)
    print("Preprocessed Data saved to preprocessed_data.csv file")

    return final_df

def create_sample(ddf: pd.DataFrame, frac=0.1, output="sample.csv"):
    separator("Sampling Data")
    df_sample = ddf.sample(frac=frac, random_state=42)
    df_sample.to_csv(output, index=False)
    print(f"File Saved: {output}")
    return df_sample

def create_kmeans(df: pd.DataFrame, n_clusters=10):
    separator("KMeans Clustering")
    KMEANS_OUTPUT = "kmeans.csv"

    # 1. Αφαίρεση NaN από αριθμητικές στήλες
    df = df.dropna(axis=1)
    
    # 2. Διαχωρισμός των στηλών Label και Traffic Type (αν υπάρχουν)
    extra_cols = []
    for col in ['Label', 'Traffic Type']:
        if col in df.columns:
            extra_cols.append(col)
    extra_data = df[extra_cols].copy() if extra_cols else None

    # 3. Επιλογή αριθμητικών δεδομένων
    df_numeric = df.select_dtypes(include="number").copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # 4. KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_numeric["Cluster"] = kmeans.fit_predict(X_scaled)

    # 5. Εύρεση των πιο αντιπροσωπευτικών σημείων (closest to centroids)
    centroids = []
    extra_rows = []
    for i in range(n_clusters):
        cluster_i = df_numeric[df_numeric["Cluster"] == i].drop("Cluster", axis=1)
        center = kmeans.cluster_centers_[i]
        closest_idx = ((cluster_i - center) ** 2).sum(axis=1).idxmin()
        centroids.append(cluster_i.loc[closest_idx])

        # Αν υπάρχουν, πάρε τις αντίστοιχες Label/Traffic Type τιμές
        if extra_data is not None:
            extra_rows.append(extra_data.iloc[closest_idx])

    # 6. Δημιουργία τελικού dataframe
    kmeans_df = pd.DataFrame(centroids)
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows).reset_index(drop=True)
        kmeans_df = pd.concat([kmeans_df.reset_index(drop=True), extra_df], axis=1)

    # 7. Αποθήκευση
    kmeans_df.to_csv(KMEANS_OUTPUT, index=False)
    print(f"✅ KMeans saved to {KMEANS_OUTPUT} with {len(kmeans_df)} representative points")

    return kmeans_df

# Finding the best K using Silhouette Score
def find_best_k_silhouette(df: pd.DataFrame, k_range=range(2, 15)):
    separator("🔍 Finding Best K using Silhouette Score")
    df = df.dropna(axis=1)
    # Επιλογή αριθμητικών και αφαίρεση NaN
    df_numeric = df.select_dtypes(include="number")
    
    if df_numeric.empty:
        print("⚠ Δεν υπάρχουν αριθμητικά δεδομένα για clustering.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"K = {k}: Silhouette Score = {score:.4f}")

    best_k = k_range[scores.index(max(scores))]
    print(f"\n✅ Βέλτιστο K με βάση το Silhouette Score: {best_k}")

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

def run_hdbscan(df, min_cluster_size=50, sample_frac=0.1):
    print("🔍 Running PCA + HDBSCAN...")

    # 1. Αφαίρεση στηλών με NaN
    df = df.dropna(axis=1)

    # 2. Διαχωρισμός των κατηγορικών (Label, Traffic Type)
    extra_cols = []
    for col in ['Label', 'Traffic Type']:
        if col in df.columns:
            extra_cols.append(col)
    extra_data = df[extra_cols].copy() if extra_cols else None

    # 3. Επιλογή αριθμητικών δεδομένων
    df_numeric = df.select_dtypes(include="number")
    print(f"Numeric shape before sampling: {df_numeric.shape}")

    # 4. Δειγματοληψία
    df_numeric = df_numeric.sample(frac=sample_frac, random_state=42)
    if extra_data is not None:
        extra_data = extra_data.loc[df_numeric.index].copy()
    print(f"Sampled shape: {df_numeric.shape}")

    # 5. Κλιμάκωση
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # 6. PCA για 90% variance
    pca = PCA(n_components=0.9, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA shape: {X_pca.shape}")

    # 7. HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
    labels = clusterer.fit_predict(X_pca)

    # 8. Προσθήκη αποτελεσμάτων
    df_result = df_numeric.copy()
    df_result["Cluster"] = labels

    if extra_data is not None:
        df_result = pd.concat([df_result.reset_index(drop=True), extra_data.reset_index(drop=True)], axis=1)

    df_result.to_csv("hdbscan_pca_sampled.csv", index=False)
    print("✅ HDBSCAN completed and saved to hdbscan_pca_sampled.csv")

    return df_result


if __name__ == "__main__":
    # df = load_data("data.csv")
    df = pd.read_csv("preprocessed_no_nulls.csv")
    # has_missing_values(df) # This dataset has no missing values
    
    # df = preprocessing(df)
    
    # df = pd.read_csv("preprocessed_data.csv")
    # df = df.dropna(how='all')  # Drop rows with all NaN values
    
    # create_sample(df, frac=0.2)
    # best_k = find_best_k_silhouette(df, k_range=range(150, 155))
    # best_k = 150 # I found that best_k is 11 from previous runs
    # create_kmeans(df, n_clusters=best_k)
    run_hdbscan(df)

