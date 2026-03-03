from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic waveform dataset for smoke tests.")
    parser.add_argument("--output-dir", default="data/mock_npz", help="Directory to write .npz files.")
    parser.add_argument("--csv", default="data/mock_concat_waveform_new.csv", help="Output CSV path.")
    parser.add_argument("--samples-per-class", type=int, default=20, help="Samples per label class.")
    parser.add_argument("--length", type=int, default=3600, help="Waveform length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    labels = ["A", "B", "Noise"]

    for label in labels:
        for i in range(args.samples_per_class):
            fname = f"{label.lower()}_{i:03d}.npz"
            t = np.linspace(0, 1, args.length)
            base = rng.normal(0, 0.15, size=(args.length, 3)).astype(np.float32)

            if label == "A":
                signal = np.sin(2 * np.pi * 8 * t)[:, None] * np.array([[1.0, 0.7, 0.4]], dtype=np.float32)
                itp = 1600 + int(rng.integers(-100, 100))
                its = itp + 120
                wave = base + signal
            elif label == "B":
                signal = np.sign(np.sin(2 * np.pi * 5 * t))[:, None] * np.array([[0.8, 0.5, 0.3]], dtype=np.float32)
                itp = 1700 + int(rng.integers(-120, 120))
                its = itp + 150
                wave = base + signal
            else:
                itp = 1500
                its = 0
                wave = base

            np.savez(out_dir / fname, data=wave.astype(np.float32), itp=np.int64(itp), its=np.int64(its))
            rows.append({"fname": fname, "label": label, "its": np.nan if label == "B" else float(its)})

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path)

    print(f"Wrote {len(df)} rows to {csv_path}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
