from datetime import datetime
# import numpy as np
# import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    # Both x and y (the features and the target) are contained in dataset.data as a dataframe.
    # To identify the target (y), use the datashape.target object, which has a name property. Use this property to index the aforementioned dataframe.
    # To identify the features (x), use the datashape.features list of object. Similarly each object in this list has a name property to index the dataframe.

    target_name = datashape.target.name
    feature_names = [f.name for f in datashape.features]

    # print(target_name)
    # print(feature_names)

    # Inspect FunctionalModel definition to identify the function to use to compute the model predictions.
    x_df = dataset.data[
        feature_names
    ]  # <class 'pandas.core.frame.DataFrame'> (1000, 28)
    y_df = dataset.data[target_name]  # <class 'pandas.core.frame.DataFrame'> (1000,)

    # print(type(x_df))
    # print(x_df.shape)
    # print(y_df.shape)

    # If this takes too many resources, feel free to limit the dataset to the first 10,000 examples.
    if len(x_df) > 10_000:
        x = x_df.iloc[:10_000]
        y = y_df.iloc[:10_000]

    x = x_df.values  # <class 'numpy.ndarray'> (1000, 28)
    y = y_df.values  # <class 'numpy.ndarray'> (1000,)

    # print(type(x))
    # print(x.shape)
    # print(y.shape)

    # Use the y (from the dataset.data) and the prediction to cumpute the accuracy.
    y_pred = functional_model.predict_class(x)  # <class 'numpy.ndarray'> (1000,)

    # print(type(y_pred))
    # print(y_pred.shape)
    # print(y[:10])
    # print(y[:10])
    # print(y_pred[:10])

    accuracy = y_pred == y  # <class 'numpy.ndarray'> (1000,)
    accuracy_value = accuracy.mean()

    # print(accuracy.shape)
    # print(type(accuracy))

    # Below is a placeholder that allows pytest to pass.
    # accuracy_value = 0.99

    current_time = datetime.now()
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]
