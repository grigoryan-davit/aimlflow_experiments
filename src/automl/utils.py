from typing import Any, Dict, Tuple, Union

import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch


def train_baseline_regression(data_module: pl.LightningDataModule) -> ElasticNet:
    baseline = make_pipeline(StandardScaler(), ElasticNet(max_iter=1000, tol=1e-3))
    baseline.fit(
        data_module.train_df.drop(columns="target"), data_module.train_df["target"]
    )
    return baseline


def train_baseline_classification(data_module: pl.LightningDataModule) -> SGDClassifier:
    baseline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    baseline.fit(
        data_module.train_df.drop(columns="target"), data_module.train_df["target"]
    )
    return baseline


def compute_test_metrics_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def compute_test_metrics_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compare_to_baseline(
    data_module: pl.LightningDataModule,
    baseline_model: Union[ElasticNet, SGDClassifier],
    model: pl.LightningModule,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    test_df = data_module.test_df

    baseline_preds = baseline_model.predict(test_df.drop(columns=["target"]))
    with torch.no_grad():
        model.eval()
        model_preds = [
            int(pred)
            for pred in model(
                torch.FloatTensor(test_df.drop(columns=["target"]).values.tolist())
            )
        ]

    if model.task == "regression":
        return (
            compute_test_metrics_regression(
                test_df["target"].values, model_preds
                # test_df["target"].values, model_preds.cpu().detach().numpy()
            ),
            compute_test_metrics_regression(test_df["target"], baseline_preds),
        )
    else:
        return (
            compute_test_metrics_classification(
                test_df["target"].values, model_preds
                # model_preds.cpu().detach().numpy(), test_df["target"].values
            ),
            compute_test_metrics_classification(baseline_preds, test_df["target"]),
        )
