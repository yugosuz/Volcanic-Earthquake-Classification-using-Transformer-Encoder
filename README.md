# Volcanic Earthquake Classification with Transformer Encoder

This repository trains and evaluates a Transformer-based classifier for volcanic earthquake waveform data.

## Repository layout

- `train.py`: training entry point.
- `summary.py`: evaluation entry point for a saved checkpoint.
- `modules/dataset.py`: dataset loading and waveform cropping.
- `modules/model_rpr.py`: Transformer RPR classifier and checkpoint helpers.
- `modules/transformer_rpr.py`: relative positional attention blocks.
- `weights/`: pretrained/model checkpoint files.
- `stations.csv`: station metadata.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup with uv

```bash
uv sync
```

`uv sync` installs runtime and development dependencies defined in `pyproject.toml`.

## Data format

By default, scripts expect:

- CSV metadata at `data/concat_waveform_new.csv`
- waveform files under `data/` referenced by `fname` column

Each waveform `.npz` file should include:

- `data`: array shaped `(N, 3)`
- `itp`: integer index
- optional `its`: integer index

CSV should include at least:

- `fname`: `.npz` filename
- `label`: one of `A`, `B`, `Noise`

## Train

```bash
uv run train.py 32 50 10 150 -t --epochs 100 --workers 4
```

Arguments:

- positional: `num_batch window_size stride channel_size`
- `-t/--is_train`: run full training (without it, script runs one debug batch)
- `--data-csv`: metadata CSV path (default `data/concat_waveform_new.csv`)
- `--data-dir`: waveform directory (default `data`)
- `--epochs`: number of epochs
- `--workers`: DataLoader workers
- `-c/--caption`: optional run label

Outputs:

- checkpoint under `weights/`
- loss plot under `plots/`
- classification metrics printed to stdout

## Evaluate

```bash
uv run summary.py 32 50 10 150 --weight weights/your_checkpoint.pth
```

## Run tests

```bash
uv run pytest
```

## Notes

- The original input data link from the paper: https://x.gd/Ro0td
- DOI: https://doi.org/10.22541/essoar.171378786.62639546/v1

## Generate synthetic test data

You can generate a small synthetic dataset to validate the training/evaluation pipeline.

```bash
uv run python tools/generate_mock_data.py
```

Then run a debug training pass (1 epoch, 1 training batch):

```bash
uv run train.py 8 50 10 150 --data-csv data/mock_concat_waveform_new.csv --data-dir data/mock_npz --workers 0
```

Run evaluation with the latest checkpoint:

```bash
LATEST=$(ls -t weights/attn_rpr_weight_w50s10c150_*.pth | head -n 1)
uv run summary.py 8 50 10 150 --data-csv data/mock_concat_waveform_new.csv --data-dir data/mock_npz --weight "$LATEST" --workers 0
```
