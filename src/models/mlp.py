from typing import List, Literal, Union, Tuple, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn


class MLP(pl.LightningModule):
    """
    Multilayer perceptron for tabular classification and regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_states: Tuple[int] = (8, 4),
        lr: float = 1e-3,
        task: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        assert not all(
            (num_classes is None, task == "classification")
        ), "If the task is classification, num_classes should be provided"

        self.task = task
        self.lr = lr

        layers = []
        for hidden_state in hidden_states:
            linear = nn.Linear(input_size, hidden_state)
            layers.append(linear)
            layers.append(nn.ReLU())
            input_size = hidden_state

        if task == "classification":
            layers.append(nn.Linear(input_size, num_classes))
            layers.append(nn.Softmax(dim=1))
            self.loss = nn.CrossEntropyLoss()

        else:
            layers.append(nn.Linear(input_size, 1))
            self.loss = nn.MSELoss()

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        batch_idx: int,
    ):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(self(x), y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "prediction": y_pred, "label": y}

    def validation_step(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        batch_idx: int,
    ):
        x, y = batch
        y_pred = self(x, y)
        loss = self.loss(y_pred, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
