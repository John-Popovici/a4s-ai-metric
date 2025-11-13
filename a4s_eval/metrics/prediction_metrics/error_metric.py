import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


@prediction_metric(name="Regression Error metrics: MAE, MSE")
def regression_error_score_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
) -> list[Measure]:
    """
    Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE).

    This metric is for regression tasks, where the true and predicted y values are
    continuous numerical quantities.

    Parameters
    ----------
    datashape : DataShape
    model : Model
    dataset : Dataset
    y_pred_proba : np.ndarray

    Returns
    -------
    list[Measure] - A list containing two `Measure` objects:
    - "MAE": Mean Absolute Error between `y_true` and `y_pred`
    - "MSE": Mean Squared Error between `y_true` and `y_pred`

    Notes
    -----
    - Applicable primarily for regression tasks, but the Protocol
    provides `y_pred_proba`?
    - Both MAE and MSE are [0, inf), with 0 indicating perfect prediction
    - Implemented by John Popovici
    """
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = y_pred_proba.squeeze()  # (n_samples,)

    MAE_metric = Measure(
        name="MAE",
        score=float(mean_absolute_error(y_true, y_pred)),
        time=date,
    )

    MSE_metric = Measure(
        name="MSE",
        score=float(mean_squared_error(y_true, y_pred)),
        time=date,
    )

    return [MAE_metric, MSE_metric]
