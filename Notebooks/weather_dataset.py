import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class WeatherDataset(Dataset):
    def __init__(self, dataset_file, day_range, split_date, train_test="train"):
        # parse user input in the class
        self.day_range = day_range
        # read and fix dataset
        df = pd.read_csv(dataset_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        # normalize the dataset
        df_mean = df.mean()
        df_std = df.std()
        df_norm = (df - df_mean) / df_std
        # store mean and std
        self.mean = torch.tensor(df_mean.to_numpy()).reshape(1, -1)
        self.std = torch.tensor(df_std.to_numpy()).reshape(1, -1)
        # split normalized dataset based on train_test variable
        if train_test == "train":
            self.dataset = df_norm[df_norm.index < split_date]
        elif train_test == "test":
            self.dataset = df_norm[df_norm.index >= split_date]
        else:
            ValueError("train_test should be either train or test")

    def __getitem__(self, index):
        # Index a range of days
        end_index = index + self.day_range
        current_series = self.dataset.iloc[index:end_index]
        day_tensor = torch.LongTensor(current_series.index.day.to_numpy())
        month_tensor = torch.LongTensor(current_series.index.month.to_numpy())
        data_values = torch.FloatTensor(current_series.values)
        return day_tensor, month_tensor, data_values

    def __len__(self):
        return self.dataset.shape[0] - self.day_range
