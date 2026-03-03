from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

from modules.dataset import PhaseDataset
from modules.model_rpr import RPRClassifier, strip_data_parallel_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("num_batch", type=int, help="Mini-batch size.")
    parser.add_argument("window_size", type=int, help="Convolution window size.")
    parser.add_argument("stride", type=int, help="Convolution stride.")
    parser.add_argument("channel_size", type=int, help="Convolution output channels.")
    parser.add_argument(
        "--weight",
        default="weights/attn_rpr_weight_w50s10c150_1702451247.3488655.pth",
        help="Checkpoint file path.",
    )
    parser.add_argument(
        "--data-csv",
        default="data/concat_waveform_new.csv",
        help="Metadata CSV path.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing npz files.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def build_eval_loader(df: pd.DataFrame, batch_size: int, workers: int, data_dir: str) -> DataLoader:
    dataset = PhaseDataset(df, data_dir=data_dir, is_eval=True)
    split = int(0.8 * len(dataset))
    indices = np.arange(len(dataset))
    test_dataset = Subset(dataset, indices[split:])
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def collect_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for src, tgt, *_ in dataloader:
            src = src.to(device, non_blocking=True)
            logits, _ = model(src)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(torch.argmax(tgt, dim=1).tolist())

    return np.array(all_preds), np.array(all_targets)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_csv, index_col=0)
    samples = df.copy()
    samples["label"] = "Noise"
    as_df = df[df["label"] == "A"].copy()
    bs_df = df[(df["label"] == "B") & (df["its"].isnull())].copy() if "its" in df.columns else df[df["label"] == "B"].copy()
    eval_df = pd.concat([as_df, bs_df, samples]).sample(frac=1, random_state=0).reset_index(drop=True)

    test_loader = build_eval_loader(eval_df, args.num_batch, args.workers, args.data_dir)

    model = RPRClassifier(
        window_size=args.window_size,
        channel_size=args.channel_size,
        stride=args.stride,
    ).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    model.load_state_dict(strip_data_parallel_prefix(checkpoint))

    pred, tgt = collect_predictions(model, test_loader, device)
    print(f"Confusion matrix:\n{confusion_matrix(tgt, pred)}")
    print(f"Accuracy: {accuracy_score(tgt, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(tgt, pred):.4f}")
    print(f"Precision: {precision_score(tgt, pred, average=None, zero_division=0)}")
    print(f"Recall: {recall_score(tgt, pred, average=None, zero_division=0)}")
    print(f"F1: {f1_score(tgt, pred, average=None, zero_division=0)}")


if __name__ == "__main__":
    main()
