"""
Loan Approval Prediction - Model Training Pipeline
Author: Behruz
Description:
    Preprocesses data, trains classification algorithms (Logistic Regression,
    Random Forest, Support Vector Machine), evaluates performance metrics,
    and saves serialized models for deployment.
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def load_and_preprocess_data(data_path: str):
    """Loads dataset and performs imputation and categorical encoding."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    print(f"[*] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # 1. Missing values imputation
    # Mode for categorical
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Median for numerical
    num_cols = ['LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Format Dependents ('3+' -> '3')
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '3')

    # 2. Categorical Label Encoding
    encoders = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']

    return X, y, encoders


def train_models(X_train, y_train, models_dir: str):
    """Trains classification models and serializes them."""
    os.makedirs(models_dir, exist_ok=True)
    trained_models = {}

    print("\n[*] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    trained_models["Logistic Regression"] = lr

    print("[*] Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    trained_models["Random Forest"] = rf

    # Save primary production model
    rf_model_path = os.path.join(models_dir, "loan_approval_model.pkl")
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"    -> Saved production model to: {rf_model_path}")

    lr_model_path = os.path.join(models_dir, "logistic_regression.pkl")
    with open(lr_model_path, "wb") as f:
        pickle.dump(lr, f)

    return trained_models


def evaluate(models, X_test, y_test):
    """Evaluates and prints classification performance."""
    print("\n" + "=" * 55)
    print(f"{'MODEL PERFORMANCE EVALUATION':^55}")
    print("=" * 55)

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Train Loan Approval Prediction Models")
    parser.add_argument("--data", type=str, default="data/loan_data.csv", help="Path to CSV dataset")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Preprocess
    X, y, encoders = load_and_preprocess_data(args.data)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"[*] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Train
    models = train_models(X_train, y_train, args.models_dir)

    # Evaluate
    evaluate(models, X_test, y_test)
    print("\n[✔] Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
