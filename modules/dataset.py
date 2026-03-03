from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


def _build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(categories=[["A", "B", "Noise"]], sparse_output=False)
    except TypeError:
        return OneHotEncoder(categories=[["A", "B", "Noise"]], sparse=False)


def random_crop(
    vals: np.ndarray,
    includes: int,
    length: int = 3000,
    noise: bool = False,
    random: bool = True,
) -> tuple[np.ndarray, int]:
    """Crop a waveform window and return the cropped values with its start index."""

    if noise:
        return vals[:length], 0

    if random:
        upper = int(np.clip(includes - 150, 1, len(vals) - length + 1))
        lower = int(np.clip(includes - 1000, 0, upper - 1))
        rand = np.random.randint(lower, upper)
        return vals[rand : rand + length], rand

    start = int(np.clip(includes - 500, 0, len(vals) - length))
    return vals[start : start + length], start


class PhaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str | Path = "data", is_eval: bool = False):
        self.data_dir = Path(data_dir)
        self.paths = df["fname"].reset_index(drop=True)
        self.labels = df["label"].reset_index(drop=True)
        self.is_eval = is_eval

        ohe = _build_ohe()
        encoded = ohe.fit_transform(self.labels.to_numpy().reshape(-1, 1))
        encoded_df = pd.DataFrame(encoded, columns=["A", "B", "Noise"])
        self.df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    def __getitem__(self, index: int):
        path = self.data_dir / self.paths.iloc[index]
        data = np.load(path, allow_pickle=True)

        wave = data["data"]
        itp = int(data["itp"])
        wave = (wave - np.mean(wave, axis=0)) / (np.std(wave, axis=0) + 1e-8)

        is_noise = self.labels.iloc[index] == "Noise"
        wave, start = random_crop(wave, itp, noise=is_noise, random=not self.is_eval)
        label = self.df.loc[index, ["A", "B", "Noise"]].values.astype(np.float32)

        if self.is_eval:
            its = int(data["its"]) - start if "its" in data.files else 0
            return wave.astype(np.float32), label, itp - start, its, str(self.paths.iloc[index])

        return wave.astype(np.float32), label

    def __len__(self) -> int:
        return len(self.paths)
