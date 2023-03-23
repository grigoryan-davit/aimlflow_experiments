"""Breast cancer dataset"""
import lightning.pytorch as pl
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class BCDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        self.data = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        label = row[self.target]
        features = torch.FloatTensor(row.drop(columns=self.target).to_list())

        return features, label


class BCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        df, target = load_breast_cancer(as_frame=True, return_X_y=True)
        self.input_size = len(df.columns)
        df["target"] = target

        self.train_df, self.val_df = train_test_split(df, test_size=0.2)
        self.train_df, self.test_df = train_test_split(self.train_df, test_size=0.2)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self) -> None:
        self.train_dataset = BCDataset(self.train_df, "target")
        self.val_dataset = BCDataset(self.val_df, "target")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
