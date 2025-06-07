# exercise 2
# up1092678
# ECE Department - University of Patras
# Data Mining and Learning Algorithms - Spring Semester 2024-2025

import dask.dataframe as dd
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
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

    # Remove columns with less than 0.01 variance
    selector = VarianceThreshold(threshold=0.01)
    try:
        X_reduced = selector.fit_transform(numeric_df)
        selected_columns = numeric_df.columns[selector.get_support()]
        removed_low_variance = list(set(numeric_df.columns) - set(selected_columns))
        if removed_low_variance:
            print(f"Remove due to low variance: {removed_low_variance}")
        numeric_df = pd.DataFrame(X_reduced, columns=selected_columns)
    except ValueError as e:
        print(f"Error in VarianceThreshold: {e}")

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

    # Remove NaN values if they exist
    df = df.dropna(axis=1)
    
    # Seperate Label and Traffic Type if they exist
    extra_cols = []
    for col in ['Label', 'Traffic Type']:
        if col in df.columns:
            extra_cols.append(col)
    extra_data = df[extra_cols].copy() if extra_cols else None

    # Select only numeric columns
    df_numeric = df.select_dtypes(include="number").copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_numeric["Cluster"] = kmeans.fit_predict(X_scaled)

    # Find centroids
    centroids = []
    extra_rows = []
    for i in range(n_clusters):
        cluster_i = df_numeric[df_numeric["Cluster"] == i].drop("Cluster", axis=1)
        center = kmeans.cluster_centers_[i]
        closest_idx = ((cluster_i - center) ** 2).sum(axis=1).idxmin()
        centroids.append(cluster_i.loc[closest_idx])

        # If extra data exists, append the corresponding row
        if extra_data is not None:
            extra_rows.append(extra_data.iloc[closest_idx])

    # Create DataFrame for centroids
    kmeans_df = pd.DataFrame(centroids)
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows).reset_index(drop=True)
        kmeans_df = pd.concat([kmeans_df.reset_index(drop=True), extra_df], axis=1)

    # Save KMeans results
    kmeans_df.to_csv(KMEANS_OUTPUT, index=False)
    print(f"KMeans saved to {KMEANS_OUTPUT} with {len(kmeans_df)} representative points")

    return kmeans_df

# Finding the best K using Silhouette Score
def find_best_k_silhouette(df: pd.DataFrame, k_range=range(2, 15)):
    separator("Finding Best K using Silhouette Score")
    df = df.dropna(axis=1)
    # Sample the data for performance
    df = df.sample(frac=0.01, random_state=42)
    # Select only numeric columns
    df_numeric = df.select_dtypes(include="number")
    if df_numeric.empty:
        print("No numeric columns for clustering.")
        return None

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # If k_range is a single value, run KMeans for that value
    if len(k_range) == 1:
        k = k_range.start
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score for K = {k}: {score:.4f}")
        return score

    # Find the best K using Silhouette Score
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"K = {k}: Silhouette Score = {score:.4f}")

    best_k = k_range[scores.index(max(scores))]
    print(f"\nBest K with highest Silhouette Score: {best_k}")

    # Graph
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
    print("Running HDBSCAN...")

    # Remove NaN values from numeric columns
    df = df.dropna(axis=1)

    # Seperate Label and Traffic Type
    extra_cols = []
    for col in ['Label', 'Traffic Type']:
        if col in df.columns:
            extra_cols.append(col)
    extra_data = df[extra_cols].copy() if extra_cols else None

    # Select numeric columns
    df_numeric = df.select_dtypes(include="number")
    print(f"Numeric shape before sampling: {df_numeric.shape}")

    # Sample the data (large dataset)
    df_numeric = df_numeric.sample(frac=sample_frac, random_state=42)
    if extra_data is not None:
        extra_data = extra_data.loc[df_numeric.index].copy()
    print(f"Sampled shape: {df_numeric.shape}")

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # PCA for 90% variance
    pca = PCA(n_components=0.9, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA shape: {X_pca.shape}")

    # Apply hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
    labels = clusterer.fit_predict(X_pca)

    # Add cluster labels to the original DataFrame
    df_result = df_numeric.copy()
    df_result["Cluster"] = labels

    if extra_data is not None:
        df_result = pd.concat([df_result.reset_index(drop=True), extra_data.reset_index(drop=True)], axis=1)

    df_result.to_csv("hdbscan_pca_sampled.csv", index=False)
    print("HDBSCAN completed and saved to hdbscan_pca_sampled.csv")

    return df_result


if __name__ == "__main__":
    df = load_data("data.csv")
    # df = pd.read_csv("preprocessed_data.csv")
    # has_missing_values(df) # This dataset has no missing values
    
    df = preprocessing(df)
    
    # df = pd.read_csv("preprocessed_data.csv")
    # df = df.dropna(how='all')  # Drop rows with all NaN values
    
    # create_sample(df, frac=0.2)
    # best_k = find_best_k_silhouette(df, k_range=range(900, 901))
    best_k = 850 # I found that best_k is 11 from previous runs
    create_kmeans(df, n_clusters=best_k)
    run_hdbscan(df)

