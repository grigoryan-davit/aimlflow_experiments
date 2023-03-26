import sys
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import mlflow.sklearn
import pandas as pd
import torch

from src.data.bc_dataset import BCDataModule
from src.models.mlp import MLP
from src.automl.utils import (
    train_baseline_regression,
    train_baseline_classification,
    compare_to_baseline
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

        args = {
            "experiment_name": sys.argv[1] if len(sys.argv) > 1 else "mlp",
            "lr": sys.argv[2] if len(sys.argv) > 2 else 1e-3,
            "batch_size": sys.argv[3] if len(sys.argv) > 3 else 16,
            "num_workers": sys.argv[4] if len(sys.argv) > 4 else 2,
            "num_epochs": sys.argv[5] if len(sys.argv) > 5 else 3,
        }

        mlflow.log_param("lr", args["lr"])
        mlflow.log_param("batch_size", args["batch_size"])
        mlflow.log_param("num_workers", args["num_workers"])
        mlflow.log_param("num_epochs", args["num_epochs"])

        data = BCDataModule(batch_size=args["batch_size"], num_workers=args["num_workers"])
        baseline = train_baseline_classification(data)

        model = train_pl(
            experiment_name=args["experiment_name"],
            model=MLP(input_size=data.input_size, lr=args["lr"], num_classes=len(data.train_df["target"].unique())),
            data_module=data,
            num_epochs=args["num_epochs"],
        )
        
        test_metrics, baseline_metrics = compare_to_baseline(data, baseline, model)

        print(test_metrics)
        print(baseline_metrics)
        for metric in test_metrics.keys():
            mlflow.log_metric(metric + "_baseline", baseline_metrics[metric])
            mlflow.log_metric(metric + "_test", test_metrics[metric])
