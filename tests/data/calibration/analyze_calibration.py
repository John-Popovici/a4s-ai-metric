"""
Calibration metric data gathering for presentation.

Computes calibration metrics and plots graphs.
Takes as argument a CSV.

By John Popovici.
"""

import argparse
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import Namespace

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


def present_calibration(data_file: str) -> None:
    dir_path: str = "tests/data/calibration/"

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
    ECE, MCE = classification_calibration_score_metric(
        dummy_data_shape, dummy_model, dummy_dataset, y_pred_proba, n_bins, dir_path
    )

    print(f"ECE={ECE}")
    print(f"MCE={MCE}")


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
    # uv run tests/data/calibration/analyze_calibration.py --csv <file-name>
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        dest="file_name",
        required=True,
        help="CSV file name",
    )
    args: Namespace = parser.parse_args()

    present_calibration(args.file_name)
