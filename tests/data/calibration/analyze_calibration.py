"""
Calibration metric data gathering for presentation.

Makes use of the metric and metric testing files:
- a4s_eval.metrics.prediction_metrics.calibration_metric
- tests.metrics.prediction_metrics.test_calibration_metric

Generates datasets. Computes metrics. Generates graphs.

By John Popovici.
"""

import datetime
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model, FeatureType, Feature
from a4s_eval.metrics.prediction_metrics.calibration_metric import (
    classification_calibration_score_metric,
)


def generate_expected_formats(y_true: np.ndarray) -> Dataset:
    date = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value=0,
        max_value=0,
    )

    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=5.0,
    )

    data_shape = DataShape(features=[], date=date, target=target)

    dates = pd.date_range("2025-01-01", periods=len(y_true), freq="D")
    df: pd.DataFrame = pd.DataFrame(
        {
            data_shape.target.name: y_true,
            data_shape.date.name: dates,
        }
    )

    dummy_dataset = Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=df,
    )

    ref_data = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [5.0, 6.0, 7.0, 8.0, 9.0],
            "target": [0, 0, 1, 1, 0],
        }
    )
    # Training dataset does not have date column
    # data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    ref_dataset = Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=ref_data,
    )

    model = Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )

    return data_shape, dummy_dataset, model


def present_calibration() -> None:
    dir_path: str = "tests/data/calibration/"

    # Run tabpfn
    # data_file: str = run_tabpfn_iris(dir_path)
    # data_file: str = run_tabpfn_prima(dir_path)
    data_file: str = "data_iris_class.csv"
    # data_file: str = "data_prima_diabetes_class.csv"

    # Generate or select CSV
    # Choose which csv to generate (if any)
    # data_file: str = generate_toy_dataset_csv(dir_path)
    # data_file: str = generate_perfect_dataset_csv(dir_path)
    # data_file: str = generate_almost_perfect_dataset_csv(dir_path)
    # data_file: str = "data_perf.csv"
    # data_file: str = "data_almost_perf.csv"
    

    y_pred_proba, y_true = read_class_csv(dir_path + data_file)

    n_bins = 10
    run_metric(y_pred_proba, y_true, n_bins, dir_path)

    graph_calibration(dir_path, data_file)


def graph_calibration(dir_path: str, data_source: str) -> None:
    data_file: str = "calibration_data.csv"
    file_path: str = dir_path + data_file

    # Print information
    print(f"Reading CSV {file_path}")

    # Read data from CSV
    df = pd.read_csv(dir_path + data_file)

    # Plot calibration curve (confidence vs accuracy)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], color="grey", linestyle="dashed")
    plt.plot(df["bin_confidence"], df["bin_accuracy"], marker="o")
    plt.xlabel("Bin Confidence")
    plt.ylabel("Bin Accuracy")
    plt.title("Calibration Curve")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f"{dir_path}figures/{data_source}.png")


def run_metric(y_pred_proba, y_true, n_bins, dir_path) -> None:
    # Print information
    print(f"Running Calibration metrics, printing to {dir_path}")

    # Generate expected formats
    dummy_data_shape, dummy_dataset, dummy_model = generate_expected_formats(y_true)

    # Run metrics
    metrics = classification_calibration_score_metric(
        dummy_data_shape, dummy_model, dummy_dataset, y_pred_proba, n_bins, dir_path
    )


def run_tabpfn_iris(dir_path: str) -> str:
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

    return file_name_out


def run_tabpfn_prima(dir_path: str) -> str:
    # from https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    file_name_in: str = "data_prima_diabetes.csv"
    file_name_out: str = "data_prima_diabetes_class.csv"

    df: pd.DataFrame = pd.read_csv(dir_path + file_name_in)

    # Extract features
    feature_cols = ["atr_1", "atr_2", "atr_3", "atr_4", "atr_5", "atr_6", "atr_7", "atr_8"]
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

    return file_name_out


def run_tabpfn(X_train, y_train, X_test) -> np.ndarray:
    print("Running TabPFN Classifier")

    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)


def generate_toy_dataset_csv(dir_path: str) -> str:
    file_name: str = "data_toy.csv"

    # Write toy data
    y_pred_proba = np.array(
        [
            [0.78, 0.22],
            [0.36, 0.64],
            [0.08, 0.92],
            [0.58, 0.42],
            [0.49, 0.51],
            [0.85, 0.15],
            [0.30, 0.70],
            [0.63, 0.37],
            [0.17, 0.83],
        ]
    )
    y_true = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1])

    # Write to CSV
    write_class_csv(dir_path + file_name, y_pred_proba, y_true)
    return file_name


def generate_perfect_dataset_csv(dir_path: str) -> str:
    file_name: str = "data_perf.csv"

    all_probs = []
    all_true = []

    for num_correct in range(1, 10, 1):  # 1 -> 9
        conf = num_correct / 10  # 1 correct means confidence of 0.1

        for num_entry in range(10):
            all_probs.append([round(conf, 2), round(1 - conf, 2)])
            if num_entry < num_correct:
                all_true.append(0)
            else:
                all_true.append(1)

    # Write to CSV
    y_pred_proba = np.array(all_probs)
    y_true = np.array(all_true)

    write_class_csv(dir_path + file_name, y_pred_proba, y_true)
    return file_name


def generate_almost_perfect_dataset_csv(dir_path: str) -> str:
    file_name: str = "data_almost_perf.csv"

    all_probs = []
    all_true = []

    for num_correct in range(1, 10, 1):  # 1 -> 9
        for num_entry in range(10):
            conf = num_correct / 10  # 1 correct means confidence of 0.1
            conf += np.random.normal(0, 0.15)
            conf = np.clip(conf, 0.01, 0.99)

            all_probs.append([conf, 1 - conf])
            if num_entry < num_correct:
                all_true.append(0)
            else:
                all_true.append(1)

    # Write to CSV
    y_pred_proba = np.array(all_probs)
    y_true = np.array(all_true)

    write_class_csv(dir_path + file_name, y_pred_proba, y_true)
    return file_name


def write_class_csv(file_path: str, y_pred_proba: np.ndarray, y_true: np.ndarray) -> None:
    # Print information
    print(f"Writing CSV {file_path}")

    # Write data to CSV
    num_classes = y_pred_proba.shape[1]
    data = {}

    # Add probability columns
    for k in range(num_classes):
        data[f"prob_{k}"] = y_pred_proba[:, k]

    # Add target column
    data["true_class"] = y_true

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def read_class_csv(file_path: str):
    # Print information
    print(f"Reading CSV {file_path}")

    # Read data from CSV
    df = pd.read_csv(file_path)

    # Detect columns starting with "prob_"
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[1]))
    y_pred_proba = df[prob_cols].to_numpy()

    # Detect column with "true_class"
    y_true = df["true_class"].to_numpy()

    return y_pred_proba, y_true


if __name__ == "__main__":
    # uv run tests/data/calibration/analyze_calibration.py
    present_calibration()
