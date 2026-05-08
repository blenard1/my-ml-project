from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data_loader import image_to_tensor, make_dataloaders
from metrics import MetricAverager, binary_metrics_from_logits, saliency_loss
from sod_model import build_model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = build_model(
        variant=config.get("variant", "baseline"),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return array


def tensor_to_mask(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().clamp(0, 1).numpy()


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = np.zeros_like(image)
    heat[..., 0] = 1.0
    heat[..., 1] = 0.18
    heat[..., 2] = 0.08
    mask_3 = np.expand_dims(mask, axis=-1)
    overlay = image * (1 - alpha * mask_3) + heat * (alpha * mask_3)
    return np.clip(overlay, 0, 1)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    metrics = MetricAverager()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model.forward_logits(images)
        loss = saliency_loss(logits, masks)
        batch_metrics = {"loss": float(loss.detach().cpu())}
        batch_metrics.update(binary_metrics_from_logits(logits, masks, threshold=threshold))
        metrics.update(batch_metrics, n=images.size(0))
    return metrics.averages()


@torch.no_grad()
def save_visualization_grid(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_path: str | Path,
    max_items: int = 4,
    threshold: float = 0.5,
) -> None:
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    logits = model.forward_logits(images)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    n_items = min(max_items, images.size(0))
    fig, axes = plt.subplots(n_items, 4, figsize=(12, 3 * n_items))
    if n_items == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n_items):
        image = tensor_to_image(images[row])
        gt = tensor_to_mask(masks[row])
        pred = tensor_to_mask(preds[row])
        overlay = make_overlay(image, pred)

        panels = [
            ("Input", image),
            ("Ground truth", gt),
            ("Predicted", pred),
            ("Overlay", overlay),
        ]
        for col, (title, panel) in enumerate(panels):
            axes[row, col].imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=1)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int = 128,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    original = image.convert("RGB")
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = image_to_tensor(resized).unsqueeze(0).to(device)
    start = time.perf_counter()
    logits = model.forward_logits(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    if threshold is not None:
        prob = (prob >= threshold).astype(np.float32)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    overlay = make_overlay(image_array, prob)
    return prob, overlay, elapsed_ms


def write_metrics(metrics: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained saliency model.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=None)
    parser.add_argument("--max-visuals", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    image_size = args.image_size or int(config.get("image_size", 128))

    _, _, test_loader, sizes = make_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        synthetic_samples=args.synthetic_samples,
    )
    print(f"Dataset sizes: {sizes}")
    metrics = evaluate_loader(model, test_loader, device, threshold=args.threshold)
    metrics["threshold"] = args.threshold
    write_metrics(metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))
    save_visualization_grid(
        model=model,
        loader=test_loader,
        device=device,
        output_path=args.output_dir / "prediction_grid.png",
        max_items=args.max_visuals,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
