from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import make_dataloaders
from evaluate import evaluate_loader, load_model_from_checkpoint


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep saliency thresholds and report precision/recall tradeoffs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ecssd"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/threshold_sweep.csv"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    _, val_loader, test_loader, sizes = make_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Dataset sizes: {sizes}")

    rows: list[dict[str, float | str]] = []
    for split_name, loader in [("validation", val_loader), ("test", test_loader)]:
        for threshold in threshold_values(args.min_threshold, args.max_threshold, args.step):
            metrics = evaluate_loader(model, loader, device, threshold=threshold)
            row: dict[str, float | str] = {"split": split_name, "threshold": threshold}
            row.update(metrics)
            rows.append(row)
            print(
                f"{split_name:10s} th={threshold:.2f} "
                f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} iou={metrics['iou']:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
