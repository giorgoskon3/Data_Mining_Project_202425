import dask.dataframe as dd
import pandas as pd
import os

# Read CSV file with Dask
df = dd.read_csv("data.csv", blocksize="16MB", dtype="object", assume_missing=True)

# Preprocessing
def preprocessing():
    nunique = df.nunique().compute()
    cols_to_drop = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Αφαιρέθηκαν {len(cols_to_drop)} σταθερές στήλες: {cols_to_drop}")

    df = df.drop(columns=["Flow ID", "Timestamp"], errors="ignore")

    df_sample = df.sample(frac=0.05).compute()
    df_sample = df_sample.convert_dtypes()

    # Save the sample DataFrame to a CSV file
    os.makedirs("compressed_datasets", exist_ok=True)
    df_sample.to_csv("compressed_datasets/sample_random.csv", index=False)
    print("Αποθηκεύτηκε: sample_random.csv")

if __name__ == "__main__":
    preprocessing()
    print("Το Preprocessing ολοκληρώθηκε.")