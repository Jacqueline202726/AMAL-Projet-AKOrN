import os
import os.path as osp
import numpy as np
import pandas as pd
import torch


def convert_onehot_to_int(X):
    # [B, H, W, 9]->[B, H, W]
    is_input = X.sum(dim=-1)
    return (is_input * (X.argmax(-1) + 1)).to(torch.int32)


# copied from https://github.com/yilundu/ired_code_release/blob/3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545/sat_dataset.py#L17
def load_rrn_dataset(data_dir, split):
    if not osp.exists(data_dir):
        raise ValueError(
            f"Data directory {data_dir} does not exist. Run data/download-rrn.sh to download the dataset."
        )

    split_to_filename = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}

    filename = osp.join(data_dir, split_to_filename[split])
    df = pd.read_csv(filename, header=None)

    def str2onehot(x):
        x = np.array(list(map(int, x)), dtype="int64")
        y = np.zeros((len(x), 9), dtype="float32")
        idx = np.where(x > 0)[0]
        y[idx, x[idx] - 1] = 1
        return y.reshape((9, 9, 9))

    features = list()
    labels = list()
    for i in range(len(df)):
        inp = df.iloc[i][0]
        out = df.iloc[i][1]
        features.append(str2onehot(inp))
        labels.append(str2onehot(out))

    return torch.tensor(np.array(features)), torch.tensor(np.array(labels))


def load_sat_dataset(path):
    with open(os.path.join(path, "features.pt"), "rb") as f:
        X = torch.load(f)
    with open(os.path.join(path, "labels.pt"), "rb") as f:
        Y = torch.load(f)
    with open(os.path.join(path, "perm.pt"), "rb") as f:
        perm = torch.load(f)
    return X, Y, perm


class SudokuDataset:

    def __init__(self, path='./data/sudoku/', train=True):

        X, Y, _ = load_sat_dataset(path)

        is_input = X.sum(dim=3, keepdim=True).int()

        indices = torch.arange(0, 9000) if train else torch.arange(9000, 10000)

        self.X = X[indices]
        self.Y = Y[indices]
        self.is_input = is_input[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.is_input[idx]


class HardSudokuDataset:
    def __init__(self, path='./data/sudoku-rnn/', split="test"):

        X, Y = load_rrn_dataset(path, split)

        is_input = X.sum(dim=3, keepdim=True).int()

        self.X = X
        self.Y = Y
        self.is_input = is_input

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.is_input[idx]
