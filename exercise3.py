import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_and_prepare_data(filename: str, target_col: str):
    df = pd.read_csv(filename)
    
    # Balanced sampling per category
    if target_col == "Traffic Type":
        df = df.groupby(target_col).apply(lambda g: g.sample(n=min(len(g), 300), random_state=42)).reset_index(drop=True)
    
    # Remove NaN values in the target column
    df = df.dropna(subset=[target_col])

    # Characteristics
    X = df.drop(columns=[target_col])

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Target
    y = df[target_col]

    return X, y

def scale_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def train_svm(X_train, y_train):
    clf = SVC(kernel="rbf", random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_mlp(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    file_path = "hdbscan_pca_sampled.csv" # "sample.csv" for sampled data, "kmeans.csv" for k-means data

    for target in ["Label", "Traffic Type"]:
        print(f"\nEvaluating for Target: {target}")
        X, y = load_and_prepare_data(file_path, target)
        X_train, X_test, y_train, y_test = scale_and_split(X, y)

        svm_model = train_svm(X_train, y_train)
        evaluate_model(svm_model, X_test, y_test, model_name="SVM")

        mlp_model = train_mlp(X_train, y_train)
        evaluate_model(mlp_model, X_test, y_test, model_name="MLP")
