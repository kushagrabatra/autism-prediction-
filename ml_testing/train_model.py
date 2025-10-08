# train_model.py
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def find_target_column(cols):
    norm = [c.strip() for c in cols]
    candidates = ["Class/ASD Traits", "Class/ASD Traits "]
    for c in candidates:
        if c in norm:
            return cols[norm.index(c)]
    for i, c in enumerate(norm):
        lc = c.lower()
        if "class" in lc or "asd" in lc:
            return cols[i]
    return cols[-1]

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Compatible across sklearn versions (don't pass `sparse=`)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop",
        sparse_threshold=0  # ensure dense output
    )

    return preprocessor, numeric_cols, cat_cols

def train_model(file_path: str):
    file_path = str(file_path)
    if not os.path.exists(file_path):
        print(f"ERROR: dataset not found at {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    target_col = find_target_column(list(df.columns))
    print(f"Using target column: '{target_col}'")

    drop_cols = []
    if "Case_No" in df.columns:
        drop_cols.append("Case_No")

    feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print("Target classes:", le.classes_.tolist())

    preprocessor, numeric_cols, cat_cols = build_preprocessor(X)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GaussianNB": GaussianNB()
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in candidates.items():
        print(f"\nTraining {name} ...")
        try:
            pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            f1 = f1_score(y_test, preds, average="weighted")
            acc = accuracy_score(y_test, preds)
            results[name] = {"f1": f1, "acc": acc}
            fitted_pipelines[name] = pipe
            print(f"{name} -> F1: {f1:.4f}, Acc: {acc:.4f}")
        except Exception as e:
            print(f"{name} training failed: {e}")

    if not results:
        print("ERROR: No candidate models trained successfully. Check data types and preprocessing.")
        sys.exit(1)

    best_name = max(results.keys(), key=lambda n: (results[n]["f1"], results[n]["acc"]))
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f}, Acc={results[best_name]['acc']:.4f})")

    os.makedirs("model", exist_ok=True)
    model_path = Path("model") / "model.pkl"
    label_path = Path("model") / "label_encoder.pkl"
    meta_path = Path("model") / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(label_path, "wb") as f:
        pickle.dump(le, f)

    metadata = {
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "target_column": target_col,
        "target_classes": le.classes_.tolist(),
        "best_model": best_name,
        "scores": results
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        y_pred_best = best_pipe.predict(X_test)
        print("\nClassification report (best model):")
        print(classification_report(y_test, y_pred_best, target_names=le.classes_.tolist()))
    except Exception as e:
        print("Could not produce classification report:", e)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved label encoder -> {label_path}")
    print(f"Saved metadata -> {meta_path}")

if __name__ == "__main__":
    default_path = os.path.join("data", "Toddler Autism dataset July 2018.csv")
    path = default_path
    if len(sys.argv) > 1:
        path = sys.argv[1]
    train_model(path)
