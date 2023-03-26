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
            layers.append(
                nn.Linear(input_size, num_classes)
                if num_classes != 2
                else nn.Linear(input_size, 1)
            )
            self.loss = nn.CrossEntropyLoss() if num_classes != 2 else nn.BCEWithLogitsLoss()

        else:
            layers.append(nn.Linear(input_size, 1))
            self.loss = nn.MSELoss()

        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        batch_idx: int,
    ):
        x, y = batch
        # y = torch.LongTensor(y)
        y_pred = self(x)
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        loss = self.loss(y_pred.float(), y)
        # loss = self.loss(y_pred.float(), y.long())

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "prediction": y_pred, "label": y}

    def validation_step(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        batch_idx: int,
    ):
        x, y = batch
        y_pred = self(x)
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        loss = self.loss(y_pred.float(), y)
        # loss = self.loss(y_pred.float(), y.long())

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
