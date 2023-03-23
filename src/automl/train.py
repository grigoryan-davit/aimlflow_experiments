from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from src.data.bc_dataset import BCDataModule
from src.models.mlp import MLP


def train_baseline(data_module: BCDataModule) -> SGDClassifier:
    baseline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    baseline.fit(
        data_module.train_df.drop(columns="target"), data_module.train_df["target"]
    )
    return baseline


def compute_test_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def compare_to_baseline(
    data_module: BCDataModule, baseline_model: SGDClassifier, model: pl.LightningModule
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    test_df = data_module.test_df

    baseline_preds = baseline_model.predict(test_df.drop("target"))
    with torch.no_grad():
        model.eval()
        model_preds = model(torch.FloatTensor(test_df.drop(columns="target").to_list()))

    return (
        compute_test_metrics(model_preds, test_df["target"]),
        compute_test_metrics(baseline_preds, test_df["target"]),
    )


def train_pl(
    experiment_name: str,
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    num_epochs: int,
):
    mlf_logger = MLFlowLogger(experiment_name=experiment_name)
    trainer = Trainer(logger=mlf_logger, max_epochs=num_epochs)
    trainer.fit(model, data_module)

    return model


if __name__ == "__main__":
    # NOTE this is what mlflow is going to run when called from cli
    # (this file is specified as an entrypoint in MLProject)
    with mlflow.start_run():
        parser = ArgumentParser()
        parser.add_argument("--experiment_name", default="mlp")
        parser.add_argument("--lr", default=1e-3)
        parser.add_argument("--batch_size", default=16)
        parser.add_argument("--num_workers", default=4)
        parser.add_argument("--num_epochs", default=5)
        args = parser.parse_args()

        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("num_epochs", args.num_epochs)

        baseline = train_baseline()

        data = BCDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
        model = train_pl(
            experiment_name=args.experiment_name,
            model=MLP(input_size=data.input_size, lr=args.lr),
            data_module=data,
            num_epochs=args.num_epochs,
        )

        test_metrics, baseline_metrics = compare_to_baseline()
        for metric in test_metrics.keys():
            mlflow.log_param(metric + "_baseline", baseline_metrics[metric])
            mlflow.log_param(metric + "_test", test_metrics[metric])
