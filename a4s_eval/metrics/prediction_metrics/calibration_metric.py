import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


@prediction_metric(name="Classification Calibration metrics: ECE, MCE")
def classification_calibration_score_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
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
    - "ECE": Expected Calibration Error — the weighted average absolute difference
        between predicted confidence and empirical accuracy across probability bins.
    - "MCE": Maximum Calibration Error — the largest absolute difference across bins.

    Notes
    -----
    - Applies to classification models that output probabilities such as `predict_proba`
    - Both ECE and MCE are [0, 1], with 0 indicating perfect prediction
    - The calculation here uses 10 equal-width probability bins
    - For multiclass models, ECE and MCE are computed on the maximum predicted class probability
    - Implemented by John Popovici, guided by towardsdatascience.com and arxiv.org/html/2501.19047v2
    """
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    # Booleans of accuracte predictions
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracies = y_pred == y_true  # (n_samples,)

    # Max of probabilities
    confidences = np.max(y_pred_proba, axis=1)  # (n_samples,)

    # Uniform bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ls, bin_rs = bins[:-1], bins[1:]

    ece, mce = 0.0, 0.0
    for bin_l, bin_r in zip(bin_ls, bin_rs):
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
