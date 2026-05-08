from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch

from data_loader import make_dataloaders, set_seed
from evaluate import save_visualization_grid
from metrics import MetricAverager, binary_metrics_from_logits, combined_saliency_loss
from sod_model import build_model

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def progress(iterable: Iterable, desc: str) -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    loss_name: str = "bce_iou",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = MetricAverager()
    metric_meter = MetricAverager()

    for batch in progress(loader, "train" if is_train else "valid"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model.forward_logits(images)
            loss = combined_saliency_loss(logits, masks, mode=loss_name)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        loss_meter.update({"loss": float(loss.detach().cpu())}, n=batch_size)
        metric_meter.update(binary_metrics_from_logits(logits.detach(), masks, threshold=threshold), n=batch_size)

    result = loss_meter.averages()
    result.update(metric_meter.averages())
    return result


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
            "history": history,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_resume_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Resumed checkpoint {path} from epoch {checkpoint['epoch']}")
    return checkpoint


def append_log(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a saliency detection CNN from scratch.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/baseline"))
    parser.add_argument("--variant", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=["bce", "bce_iou", "bce_dice", "bce_dice_iou", "focal_dice"],
        default="bce_iou",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, choices=[3, 4], default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir) if args.data_dir else None
    config["output_dir"] = str(args.output_dir)
    config["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint else None
    config["device"] = str(device)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_loader, val_loader, _, sizes = make_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        synthetic_samples=args.synthetic_samples,
    )
    print(f"Dataset sizes: {sizes}")
    print(f"Using device: {device}")

    model = build_model(variant=args.variant, base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    last_path = args.output_dir / "checkpoint_last.pt"
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume and last_path.exists():
        checkpoint = load_resume_checkpoint(last_path, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Initialized model weights from {args.init_checkpoint}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            threshold=args.threshold,
            loss_name=args.loss,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            threshold=args.threshold,
            loss_name=args.loss,
        )
        scheduler.step(val_metrics["loss"])

        row = {"epoch": epoch}
        row["lr"] = optimizer.param_groups[0]["lr"]
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        append_log(args.output_dir / "training_log.csv", row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, config, history)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, config, history)
            print("New best model saved.")
        else:
            epochs_without_improvement += 1

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            save_visualization_grid(
                model=model,
                loader=val_loader,
                device=device,
                output_path=args.output_dir / f"val_predictions_epoch_{epoch:03d}.png",
                max_items=4,
                threshold=args.threshold,
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break


if __name__ == "__main__":
    main()
