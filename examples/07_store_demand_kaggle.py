import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """A transformer that adds calendar features to a DataFrame."""

    def __init__(self, date_col, drop_date_col=False, drop_original=True):
        """Initialize transformer with parameters.

        Args:
            date_col (str): Name of the date column.
            drop_date_col (bool): Whether to drop the original date column.
            drop_original (bool): Whether to drop the original date column.
        """
        self.date_col = date_col
        self.drop_date_col = drop_date_col
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """No fitting necessary for this transformer."""
        return self

    def encode(self, X, column):
        """Encode a column using sine and cosine transformations."""
        X[column + "_sin"] = np.sin(2 * np.pi * X[column] / X[column].max())
        X[column + "_cos"] = np.cos(2 * np.pi * X[column] / X[column].max())
        return X

    def transform(self, X):
        """Transform the DataFrame by adding calendar features."""
        X = X.copy()
        X["dow"] = X[self.date_col].dt.dayofweek
        X = self.encode(X, "dow")
        X["day"] = X[self.date_col].dt.day
        X = self.encode(X, "day")
        X["week"] = X[self.date_col].dt.isocalendar().week
        X = self.encode(X, "week")
        X["month"] = X[self.date_col].dt.month
        X = self.encode(X, "month")
        if self.drop_date_col:
            X = X.drop(columns=[self.date_col])
        if self.drop_original:
            X = X.drop(columns=["dow", "day", "week", "month"])
        return X


train = pd.read_csv(
    "store_demand_train.csv", parse_dates=True, dtype={"store": str, "item": str}
)

test = pd.read_csv(
    "store_demand_test.csv", parse_dates=True, dtype={"store": str, "item": str}
)

TRAIN_CUTOFF = pd.Timestamp("2017-10-01")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

calendar_transformer = CalendarFeatureTransformer(date_col="date")

ohe_transformer = OneHotEncoder(sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ("calendar", calendar_transformer, slice(None)),
        ("ohe", ohe_transformer, ["store", "item"]),
    ],
    remainder="passthrough",
)
ohe_transformer.fit_transform(train[["store", "item"]])

train_df = calendar_transformer.fit_transform(train)
train_df = train_df.rename_axis("time_idx").reset_index()
valid_df = train_df[train_df.date >= TRAIN_CUTOFF].copy()

test_df = calendar_transformer.transform(test)
test_df = test_df.rename_axis("time_idx").reset_index()
