import numpy as np
import pandas as pd

from pathlib import Path

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


class Logger:
    """Simple logging class."""
    def __init__(self, valid: bool, logging_file_path: str) -> None:
        self.valid: bool = valid
        if self.valid:
            self.path: Path = Path(logging_file_path)
            open(file=self.path, mode='w', encoding="utf-8").close()

    def write(self, data:str) -> None:
        if self.valid:
            with open(file=self.path, mode="a", encoding="utf-8") as f:
                f.write(data)
    
    def write_bar(self) -> None:
        self.write("="*20 + "\n")


@prediction_metric(name="Classification Calibration metrics: ECE, MCE")
def classification_calibration_score_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    logging_dir_path: str = "",
) -> list[Measure]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    This metric is for classification tasks and calculates the weighted average error
    of the estimated probabilities.

    Parameters
    ----------
    datashape : DataShape
    model : Model
    dataset : Dataset
    y_pred_proba : np.ndarray

    Returns
    -------
    list[Measure] - A list of two `Measure` objects:
    - "ECE": Expected Calibration Error is the weighted average absolute difference
        between predicted confidence and empirical accuracy across probability bins.
    - "MCE": Maximum Calibration Error is the largest absolute difference across bins.

    Notes
    -----
    - Applies to classification models that output probabilities such as `predict_proba`
    - Both ECE and MCE are [0, 1], with 0 indicating perfect prediction
    - The calculation here uses 10 equal-width probability bins
    - For multiclass models, ECE and MCE are computed on the maximum predicted class probability
    - Implemented by John Popovici, guided by towardsdatascience.com and arxiv.org/html/2501.19047v2
    """
    log: Logger = Logger(bool(logging_dir_path), logging_dir_path + "calibration_log.txt")
    log_data: Logger = Logger(bool(logging_dir_path), logging_dir_path + "calibration_data.csv")
    log.write("Running Calibration Metrics\n")
    log.write_bar()

    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    # Booleans of accuracte predictions
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracies = y_pred == y_true  # (n_samples,)

    # Max of probabilities
    confidences = np.max(y_pred_proba, axis=1)  # (n_samples,)

    # Log setup data
    log.write(f"n_bins={n_bins}\n")
    log.write(f"date={date}\n")
    log.write(f"accuracies_shape={accuracies.shape}\n")
    log.write(f"confidences_shape={confidences.shape}\n")
    log_data.write("bin_min,bin_max,bin_confidence,bin_accuracy\n")

    # Uniform bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ls, bin_rs = bins[:-1], bins[1:]

    ece, mce = 0.0, 0.0
    for bin_l, bin_r in zip(bin_ls, bin_rs):

        # Log bin data
        log.write_bar()
        log.write(f"Bin=({bin_l}, {bin_r}]\n")

        # Determine if sample is in bin
        in_bin = np.logical_and(confidences > bin_l.item(), confidences <= bin_r.item())

        if in_bin.sum() > 0:
            # Average accuracy of bin
            acc_in_bin = accuracies[in_bin].mean()
            # Average confidence in bin
            conf_in_bin = confidences[in_bin].mean()
            # Abs diff
            abs_diff = np.abs(acc_in_bin - conf_in_bin)

            ece += abs_diff * in_bin.mean()
            mce = max(mce, abs_diff)

            # Log bin data
            log.write(f"in_bin={in_bin.sum()}\n")
            log.write(f"confidence={conf_in_bin}\n")
            log.write(f"accuracy={acc_in_bin}\n")
            log.write(f"calibration_error={abs_diff}\n")
            log_data.write(f"{round(bin_l, 2)},{round(bin_r, 2)},{round(conf_in_bin, 4)},{round(acc_in_bin, 4)}\n")

    # Log total data
    log.write_bar()
    log.write(f"ECE={ece}\nMCE={mce}\n")

    ECE_metric = Measure(
        name="ECE",
        score=ece,
        time=date,
    )

    MCE_metric = Measure(
        name="MCE",
        score=mce,
        time=date,
    )

    return [ECE_metric, MCE_metric]
