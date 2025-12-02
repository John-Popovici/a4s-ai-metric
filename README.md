# Metric Implementation

This project is John Popovici's implementation of the Calibration Error Metric (ECE, MCE) into the A4S platform.

The repository can be found at `https://github.com/John-Popovici/a4s-ai-metric`

## Installation

1. Clone the repository
2. In the repository `uv sync`

## Using the metric

The metric is implemented in `a4s_eval/metrics/prediction_metrics/calibration_metric.py`
The metric tests are in `tests/metrics/prediction_metrics/test_calibration_metric.py`

```bash
uv run pytest tests/metrics/prediction_metrics/test_calibration_metric.py
```

The metric can be used as such

```python
ECE, MCE = classification_calibration_score_metric(
    data_shape, model, dataset, y_pred_proba, 
    n_bins, # optional, default value is 10
    dir_path, # optional for logging
)
```

Already implemented are some files to facilitate exploration of the metric within the `tests/data/calibration/` directory:
- Generate Predictions
    - `generate_data.py`
        - Generates toy probabilities
    - `run_tabpfn.py --data prima`
        - Generates probabilities from data using TabPFN
        - Dataset support is hard-coded for demo purposes
- Run Calibration Metric and Graph Logs
    - `analyze_calibration.py --csv data_perf.csv`

Already generated are some datasets. Graphs can be found in `tests/data/calibration/figures`
