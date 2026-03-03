from __future__ import annotations

import numpy as np
import pandas as pd

from modules.dataset import PhaseDataset, random_crop


def test_random_crop_noise_returns_zero_start():
    vals = np.arange(4000).reshape(-1, 1)
    cropped, start = random_crop(vals, includes=1000, length=3000, noise=True)
    assert start == 0
    assert cropped.shape == (3000, 1)


def test_phase_dataset_returns_expected_shapes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "sample.npz",
        data=np.random.randn(3500, 3).astype(np.float32),
        itp=1500,
        its=1700,
    )

    df = pd.DataFrame({"fname": ["sample.npz"], "label": ["A"]})

    train_ds = PhaseDataset(df, data_dir=data_dir, is_eval=False)
    wave, label = train_ds[0]
    assert wave.shape == (3000, 3)
    assert label.shape == (3,)
    assert np.isclose(label.sum(), 1.0)

    eval_ds = PhaseDataset(df, data_dir=data_dir, is_eval=True)
    wave, label, itp, its, path = eval_ds[0]
    assert wave.shape == (3000, 3)
    assert label.shape == (3,)
    assert isinstance(itp, (int, np.integer))
    assert isinstance(its, (int, np.integer))
    assert path == "sample.npz"
