import datetime
import uuid

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metrics.prediction_metrics.error_metric import (
    regression_error_score_metric,
)


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def y_pred_proba(ref_model: Model, test_dataset: Dataset) -> np.ndarray:
    session = ort.InferenceSession("./tests/data/lcld_v2_random_forest.onnx")
    df = test_dataset.data[[f.name for f in test_dataset.shape.features]]
    x_test = df.astype(np.double).to_numpy()

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[1].name
    pred_onx = session.run([label_name], {input_name: x_test})[0]
    y_pred_proba = np.array([list(d.values()) for d in pred_onx])

    return y_pred_proba


def test_model_error_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = regression_error_score_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 2
    MAE_metric: Measure = metrics[0]
    MSE_metric: Measure = metrics[1]

    assert MAE_metric.name == "MAE"
    assert MSE_metric.name == "MSE"

    for metric in [MAE_metric, MSE_metric]:
        assert isinstance(metric.score, float)
        assert 0 <= metric.score <= 1
        assert isinstance(metric.time, datetime.datetime)
