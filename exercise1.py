# exercise 1
# up1092678
# ECE Department - University of Patras
# Data Mining and Learning Algorithms - Spring Semester 2024-2025

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def separator(title: str = None):
    if title:
        print(60*"-")
        print(title)
        print(60*"-")
    else:
        print(60*"-")

def load_data(file_path: str) -> dd.DataFrame:
    try:
        df = dd.read_csv(file_path)
        separator("Data Loaded")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def show_info(df: dd.DataFrame):
    separator("Data Info")
    print(df.info())

def show_desc(df: dd.DataFrame):
    separator("Data Description")
    desc = df.describe().compute()
    print(desc)
    return desc

def show_head(df: dd.DataFrame, n: int = 5):
    separator("Data Head")
    print(df.head(n))

def find_NaNs(df: dd.DataFrame) -> pd.DataFrame:
    separator("Missing Values")
    NaNs = df.isnull().sum().compute().sort_values(ascending=True)
    print(NaNs)
    return NaNs

def show_histogram(df: dd.DataFrame, output_dir="histograms"):
    separator("Histogram for all columns")
    MIN_UNIQUE = 2
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        safe_col = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in col)
        filename = os.path.join(output_dir, f"{safe_col}_histogram.png")

        if os.path.exists(filename):
            print(f"Παράλειψη (ήδη υπάρχει): {filename}")
            continue

        print(f"\n▶ Στήλη: {col}")
        plt.figure(figsize=(8, 4))

        try:
            data = df[col].dropna().sample(frac=0.05).compute()
        except:
            data = df[col].dropna().compute()

        if pd.api.types.is_numeric_dtype(data):
            sns.histplot(data, bins=50, kde=False)
            plt.title(f'Ιστόγραμμα (numeric): {col}')
        else:
            data = data.astype("object")
            value_counts = data.value_counts()

            if value_counts.shape[0] >= MIN_UNIQUE:
                sns.barplot(x=value_counts.index[:20], y=value_counts.values[:20])
                plt.xticks(rotation=45)
                plt.title(f'Ραβδόγραμμα (categorical): {col}')
            else:
                print("Πολύ λίγες μοναδικές τιμές, παράλειψη.")
                plt.close()
                continue

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Αποθηκεύτηκε: {filename}")

    
# def show_heatmap(df: dd.DataFrame):
#     # Υπολογισμός μικρού δείγματος (1%)
#     df_sample = df.sample(frac=0.01).compute()

#     # Μετατροπή τύπων (π.χ. string[pyarrow] -> object)
#     df_sample = df_sample.convert_dtypes()

#     # Κράτα μόνο αριθμητικές στήλες
#     df_numeric = df_sample.select_dtypes(include=["number"]).dropna()

#     if df_numeric.shape[1] < 2:
#         print("✘ Δεν υπάρχουν αρκετές αριθμητικές στήλες για heatmap.")
#     else:
#         corr = df_numeric.corr()
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
#     plt.title("Heatmap of Correlation Matrix")
#     plt.tight_layout()
#     plt.show()

def show_heatmap(df: dd.DataFrame):

    separator("Heatmap Correlation Matrix")

    # Υπολογισμός δείγματος
    df_sample = df.sample(frac=0.01).compute()
    df_sample = df_sample.convert_dtypes()

    # Κρατάμε μόνο αριθμητικά
    df_numeric = df_sample.select_dtypes(include=["number"]).dropna()

    # Αν δεν έχει 2+ στήλες, δεν φτιάχνουμε heatmap
    if df_numeric.shape[1] < 2:
        print("✘ Δεν υπάρχουν αρκετές αριθμητικές στήλες για heatmap.")
        return

    # Υπολογισμός πίνακα συσχέτισης
    corr = df_numeric.corr()

    # Φιλτράρουμε τις συσχετίσεις που έχουν τιμή > 0.9 για έξτρα αναφορά (προαιρετικό)
    print("\nΙσχυρές συσχετίσεις (|corr| > 0.9):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            value = corr.iloc[i, j]
            if abs(value) > 0.9:
                print(f"{corr.columns[i]} <-> {corr.columns[j]}: {value:.2f}")

    # Δημιουργία heatmap με καλύτερη μορφοποίηση
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .6}
    )
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Correlation Heatmap of Numeric Features", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("clean_heatmap.png", dpi=300)
    plt.show()

    
if __name__ == "__main__":
    data = "data.csv"
    
    # Load the data using Dask
    df = load_data(data)
    
    # Show info
    show_info(df)
    
    # Show description
    desc = show_desc(df)
    
    # Save description to CSV
    desc.to_csv("data_desc.csv", index=True)
    
    # Show head of the data (first 5 rows)
    show_head(df)
    
    # Show histograms for all columns
    # show_histogram(df)
    
    # Show heatmap
    # show_heatmap(df)