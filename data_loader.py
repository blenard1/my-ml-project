from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_HINTS = ("mask", "gt", "ground", "truth", "saliency", "label")
IMAGE_HINTS = ("image", "img", "jpeg", "jpg", "input")


@dataclass(frozen=True)
class SaliencyPair:
    image_path: Path
    mask_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _looks_like_mask(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return any(hint in lowered for hint in MASK_HINTS)


def _looks_like_image(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return any(hint in lowered for hint in IMAGE_HINTS)


def list_image_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_pairs(root: str | Path) -> list[SaliencyPair]:
    """Discover paired RGB images and saliency masks by matching filename stems."""

    root = Path(root)
    files = list_image_files(root)
    if not files:
        return []

    mask_candidates = [p for p in files if _looks_like_mask(p)]
    image_candidates = [p for p in files if p not in mask_candidates]

    if not image_candidates:
        image_candidates = [p for p in files if _looks_like_image(p)]
        mask_candidates = [p for p in files if p not in image_candidates]

    masks_by_stem: dict[str, Path] = {}
    for mask in mask_candidates:
        masks_by_stem.setdefault(mask.stem.lower(), mask)

    pairs: list[SaliencyPair] = []
    for image in image_candidates:
        mask = masks_by_stem.get(image.stem.lower())
        if mask is not None and mask != image:
            pairs.append(SaliencyPair(image_path=image, mask_path=mask))

    return sorted(pairs, key=lambda pair: pair.image_path.name.lower())


def save_pairs_csv(pairs: Sequence[SaliencyPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "mask_path"])
        for pair in pairs:
            writer.writerow([str(pair.image_path), str(pair.mask_path)])


def load_pairs_csv(path: str | Path) -> list[SaliencyPair]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [SaliencyPair(Path(row["image_path"]), Path(row["mask_path"])) for row in reader]


def split_pairs(
    pairs: Sequence[SaliencyPair],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[list[SaliencyPair], list[SaliencyPair], list[SaliencyPair]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    n_total = len(pairs)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))
    n_train = max(0, n_total - n_val - n_test)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]
    return train_pairs, val_pairs, test_pairs


def _resize_pair(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    return (
        image.resize((size, size), Image.Resampling.BILINEAR),
        mask.resize((size, size), Image.Resampling.NEAREST),
    )


def _random_crop_pair(image: Image.Image, mask: Image.Image, min_scale: float = 0.82) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    scale = random.uniform(min_scale, 1.0)
    crop_w = max(1, int(width * scale))
    crop_h = max(1, int(height * scale))
    left = random.randint(0, max(0, width - crop_w))
    top = random.randint(0, max(0, height - crop_h))
    box = (left, top, left + crop_w, top + crop_h)
    return image.crop(box), mask.crop(box)


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < 0.7:
        image, mask = _random_crop_pair(image, mask)

    if random.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))

    if random.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))

    return image, mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    array = (array >= 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class SaliencyDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[SaliencyPair] | None = None,
        root: str | Path | None = None,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        if pairs is None:
            if root is None:
                raise ValueError("Either pairs or root must be provided.")
            pairs = discover_pairs(root)
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        if self.augment:
            image, mask = _augment_pair(image, mask)

        image, mask = _resize_pair(image, mask, self.image_size)
        return {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }


class SyntheticSaliencyDataset(Dataset):
    """Tiny generated dataset for smoke testing the full pipeline."""

    def __init__(self, length: int = 32, image_size: int = 64, seed: int = 123) -> None:
        self.length = int(length)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index)
        size = self.image_size
        image = Image.new("RGB", (size, size), color=(rng.randint(20, 80), rng.randint(20, 80), rng.randint(20, 80)))
        mask = Image.new("L", (size, size), color=0)
        image_draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)

        shape = rng.choice(["ellipse", "rectangle"])
        radius = rng.randint(size // 7, size // 4)
        cx = rng.randint(radius + 2, size - radius - 2)
        cy = rng.randint(radius + 2, size - radius - 2)
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        color = (rng.randint(140, 255), rng.randint(120, 255), rng.randint(100, 255))
        if shape == "ellipse":
            image_draw.ellipse(box, fill=color)
            mask_draw.ellipse(box, fill=255)
        else:
            image_draw.rectangle(box, fill=color)
            mask_draw.rectangle(box, fill=255)

        noise = np.random.default_rng(self.seed + index).normal(0, 8, size=(size, size, 3))
        noisy = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy, mode="RGB")

        return {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "image_path": f"synthetic_{index:04d}.png",
            "mask_path": f"synthetic_{index:04d}_mask.png",
        }


def make_dataloaders(
    data_dir: str | Path | None = None,
    image_size: int = 128,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_samples: int | None = None,
    synthetic_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    if synthetic_samples is not None:
        dataset = SyntheticSaliencyDataset(length=synthetic_samples, image_size=image_size, seed=seed)
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_fraction)))
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_train = max(1, len(indices) - n_val - n_test)
        train_ds = Subset(dataset, indices[:n_train])
        val_ds = Subset(dataset, indices[n_train : n_train + n_val])
        test_ds = Subset(dataset, indices[n_train + n_val :])
    else:
        if data_dir is None:
            raise ValueError("data_dir is required unless synthetic_samples is set.")
        pairs = discover_pairs(data_dir)
        if not pairs:
            raise FileNotFoundError(f"No image/mask pairs found under {data_dir}. Run download_ecssd.py first.")
        train_pairs, val_pairs, test_pairs = split_pairs(
            pairs,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            max_samples=max_samples,
        )
        train_ds = SaliencyDataset(train_pairs, image_size=image_size, augment=True)
        val_ds = SaliencyDataset(val_pairs, image_size=image_size, augment=False)
        test_ds = SaliencyDataset(test_pairs, image_size=image_size, augment=False)

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    return train_loader, val_loader, test_loader, sizes


def pairs_summary(pairs: Iterable[SaliencyPair]) -> str:
    pairs = list(pairs)
    return f"{len(pairs)} image/mask pairs"
