import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class WifiLTSChannelDataset(Dataset):
    """
    input 320:
      I[0], Q[0], I[1], Q[1], ..., I[159], Q[159]

    label 104:
      I[0], Q[0], I[1], Q[1], ..., I[51], Q[51]

    preprocessing:
      - drop initial 64 real values from input
      - input  -> (2, 64, 2)
      - label  -> (52, 2)
    """

    def __init__(self, csv_path: str, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

        df = pd.read_csv(csv_path, header=None)
        data = df.to_numpy(dtype=np.float32)

        expected_total_len = 320 + 104
        if data.shape[1] != expected_total_len:
            raise ValueError(
                f"Each row must have length {expected_total_len}, "
                f"but got shape {data.shape}."
            )

        x_raw = data[:, 64:320]   # shape: (N, 256)
        y_raw = data[:, 320:424]  # shape: (N, 104)

        self.x = x_raw.reshape(-1, 2, 64, 2)  # final input
        self.y = y_raw.reshape(-1, 52, 2)     # final label

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        return x, y