"""
Calibration metric data generation with TabPFN.

By John Popovici.
"""
import argparse

import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from argparse import Namespace


def run_tabpfn_iris(dir_path: str) -> None:
    # from https://archive.ics.uci.edu/dataset/53/iris
    file_name_in: str = "data_iris.csv"
    file_name_out: str = "data_iris_class.csv"

    df: pd.DataFrame = pd.read_csv(dir_path + file_name_in)

    # Extract features
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].to_numpy()
    y = df["species_id"].to_numpy()

    # Train test split
    # purposefully don't train too much to allow for more diverse outcomes
    # as well as more test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True, stratify=y, random_state=42
    )

    # Run the model
    y_pred_proba = run_tabpfn(X_train, y_train, X_test)
    write_class_csv(dir_path + file_name_out, y_pred_proba, y_test)


def run_tabpfn_prima(dir_path: str) -> None:
    # from https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    file_name_in: str = "data_prima_diabetes.csv"
    file_name_out: str = "data_prima_diabetes_class.csv"

    df: pd.DataFrame = pd.read_csv(dir_path + file_name_in)

    # Extract features
    feature_cols = [
        "atr_1",
        "atr_2",
        "atr_3",
        "atr_4",
        "atr_5",
        "atr_6",
        "atr_7",
        "atr_8",
    ]
    X = df[feature_cols].to_numpy()
    y = df["out"].to_numpy()

    # Train test split
    # purposefully don't train too much to allow for more diverse outcomes
    # as well as more test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, shuffle=True, stratify=y, random_state=42
    )

    # Run the model
    y_pred_proba = run_tabpfn(X_train, y_train, X_test)
    write_class_csv(dir_path + file_name_out, y_pred_proba, y_test)


def run_tabpfn(X_train, y_train, X_test) -> np.ndarray:
    print("Running TabPFN Classifier")

    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)


def write_class_csv(file_path: str, y_pred: np.ndarray, y_true: np.ndarray) -> None:
    # Print information
    print(f"Writing CSV {file_path}")

    # Write data to CSV
    num_classes = y_pred.shape[1]
    data = {}

    # Add probability columns
    for k in range(num_classes):
        data[f"prob_{k}"] = y_pred[:, k]

    # Add target column
    data["true_class"] = y_true

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # uv run tests/data/calibration/run_tabpfn.py --data prima
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        dest="data",
        required=True,
        help="CSV file name",
    )
    args: Namespace = parser.parse_args()

    dir_path: str = "tests/data/calibration/"

    # Hardcoded
    if args.data == "iris":
        data_file: str = run_tabpfn_iris(dir_path)
    elif args.data == "prima":
        data_file: str = run_tabpfn_prima(dir_path)
    else:
        raise Exception("Not supported dataset")
