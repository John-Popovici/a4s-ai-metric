"""
Calibration metric toy data generation.

By John Popovici.
"""
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    # uv run tests/data/calibration/generate_data.py
    dir_path: str = "tests/data/calibration/"

    generate_toy_dataset_csv(dir_path)
    generate_perfect_dataset_csv(dir_path)
    generate_almost_perfect_dataset_csv(dir_path)

