from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from modules.dataset import PhaseDataset
from modules.model_rpr import RPRClassifier, strip_data_parallel_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Transformer RPR model for volcanic earthquake classification."
    )
    parser.add_argument("num_batch", type=int, help="Mini-batch size.")
    parser.add_argument("window_size", type=int, help="Convolution window size.")
    parser.add_argument("stride", type=int, help="Convolution stride.")
    parser.add_argument("channel_size", type=int, help="Convolution output channels.")
    parser.add_argument("-t", "--is_train", action="store_true", help="Train for full epochs.")
    parser.add_argument("-c", "--caption", help="Optional run caption.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--data-csv",
        default="data/concat_waveform_new.csv",
        help="Metadata CSV path.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing npz files.")
    return parser.parse_args()


def build_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    as_df = df[df["label"] == "A"].copy()
    bs_df = df[df["label"] == "B"].copy()
    shuffled = pd.concat([as_df, bs_df, df]).sample(frac=1, random_state=0)
    return shuffled.reset_index(drop=True)


def make_loaders(df: pd.DataFrame, batch_size: int, workers: int, data_dir: str):
    dataset = PhaseDataset(df, data_dir=data_dir)
    train_size = int(0.8 * len(dataset))
    indices = np.arange(len(dataset))
    train_dataset = Subset(dataset, indices[:train_size])
    eval_dataset = Subset(dataset, indices[train_size:])

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(eval_dataset, **loader_kwargs),
        DataLoader(eval_dataset, **loader_kwargs),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    single_batch: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for src, tgt in dataloader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(src)
        loss = criterion(logits, torch.argmax(tgt, dim=1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if single_batch:
            break
    return float(np.mean(losses))


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits, _ = model(src)
            losses.append(float(criterion(logits, torch.argmax(tgt, dim=1)).item()))
    return float(np.mean(losses))


def collect_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device, non_blocking=True)
            logits, _ = model(src)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(torch.argmax(tgt, dim=1).tolist())
    return np.array(preds), np.array(targets)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.caption:
        print(args.caption)

    df = build_dataframe(args.data_csv)
    print(f"Samples: {len(df)}")
    print(df["label"].value_counts())

    train_loader, val_loader, test_loader = make_loaders(
        df, batch_size=args.num_batch, workers=args.workers, data_dir=args.data_dir
    )

    model = RPRClassifier(
        window_size=args.window_size,
        channel_size=args.channel_size,
        stride=args.stride,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-4)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs = args.epochs if args.is_train else 1

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            single_batch=not args.is_train,
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    Path("weights").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    stamp = time.time()
    weight_path = Path(
        f"weights/attn_rpr_weight_w{args.window_size}s{args.stride}c{args.channel_size}_{stamp}.pth"
    )
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, weight_path)
    print(f"Saved model: {weight_path}")

    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plot_path = Path(f"plots/loss_w{args.window_size}s{args.stride}c{args.channel_size}_{stamp}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")

    eval_model = RPRClassifier(
        window_size=args.window_size,
        channel_size=args.channel_size,
        stride=args.stride,
    ).to(device)
    loaded = torch.load(weight_path, map_location=device)
    eval_model.load_state_dict(strip_data_parallel_prefix(loaded))

    pred, tgt = collect_predictions(eval_model, test_loader, device)
    print(f"Confusion matrix:\n{confusion_matrix(tgt, pred)}")
    print(f"Accuracy: {accuracy_score(tgt, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(tgt, pred):.4f}")
    print(f"Precision: {precision_score(tgt, pred, average=None, zero_division=0)}")
    print(f"Recall: {recall_score(tgt, pred, average=None, zero_division=0)}")
    print(f"F1: {f1_score(tgt, pred, average=None, zero_division=0)}")


if __name__ == "__main__":
    main()
