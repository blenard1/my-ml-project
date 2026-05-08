from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ]
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SaliencyCNN(nn.Module):
    """CNN encoder-decoder for salient object detection, built without pretrained layers."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        batch_norm: bool = False,
        dropout: float = 0.0,
        skip_connections: bool = False,
    ) -> None:
        super().__init__()
        if depth not in {3, 4}:
            raise ValueError("depth must be 3 or 4 to match the project requirement.")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.skip_connections = skip_connections

        encoder_blocks = []
        channels = in_channels
        encoder_channels = []
        for level in range(depth):
            out_channels = base_channels * (2**level)
            encoder_blocks.append(ConvBlock(channels, out_channels, batch_norm=batch_norm, dropout=dropout))
            encoder_channels.append(out_channels)
            channels = out_channels

        self.encoder = nn.ModuleList(encoder_blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels, channels * 2, batch_norm=batch_norm, dropout=dropout)

        decoder_blocks = []
        upsamplers = []
        decoder_channels = list(reversed(encoder_channels))
        current_channels = channels * 2
        for out_channels in decoder_channels:
            upsamplers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=2, stride=2))
            block_in_channels = out_channels * 2 if skip_connections else out_channels
            decoder_blocks.append(ConvBlock(block_in_channels, out_channels, batch_norm=batch_norm, dropout=dropout))
            current_channels = out_channels

        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoder = nn.ModuleList(decoder_blocks)
        self.output = nn.Conv2d(current_channels, 1, kernel_size=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        skip_features: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            if self.skip_connections:
                skip_features.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for idx, (upsample, block) in enumerate(zip(self.upsamplers, self.decoder)):
            x = upsample(x)
            if self.skip_connections:
                skip = skip_features[-(idx + 1)]
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "baseline"
    base_channels: int = 32
    depth: int = 4
    batch_norm: bool = False
    dropout: float = 0.0
    skip_connections: bool = False


def config_for_variant(variant: str, base_channels: int = 32, depth: int = 4) -> ModelConfig:
    variant = variant.lower()
    if variant == "baseline":
        return ModelConfig(
            variant=variant,
            base_channels=base_channels,
            depth=depth,
            batch_norm=False,
            dropout=0.0,
            skip_connections=False,
        )
    if variant == "improved":
        return ModelConfig(
            variant=variant,
            base_channels=base_channels,
            depth=depth,
            batch_norm=True,
            dropout=0.05,
            skip_connections=True,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def build_model(
    variant: str = "baseline",
    base_channels: int = 32,
    depth: int = 4,
    batch_norm: bool | None = None,
    dropout: float | None = None,
    skip_connections: bool | None = None,
) -> SaliencyCNN:
    config = config_for_variant(variant, base_channels=base_channels, depth=depth)
    return SaliencyCNN(
        base_channels=config.base_channels,
        depth=config.depth,
        batch_norm=config.batch_norm if batch_norm is None else batch_norm,
        dropout=config.dropout if dropout is None else dropout,
        skip_connections=config.skip_connections if skip_connections is None else skip_connections,
    )
