from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

from evaluate import load_model_from_checkpoint, predict_image
from sod_model import build_model


def default_checkpoint_path() -> str:
    candidates = [
        Path("checkpoints/improved/best_model.pt"),
        Path("checkpoints/baseline/best_model.pt"),
        Path("checkpoints/ecssd_smoke/best_model.pt"),
        Path("checkpoints/smoke/best_model.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


@st.cache_resource
def get_model(checkpoint_path: str, variant: str, base_channels: int, depth: int, device_name: str):
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        model, config = load_model_from_checkpoint(checkpoint, device)
        return model, device, config, True

    model = build_model(variant=variant, base_channels=base_channels, depth=depth).to(device)
    model.eval()
    return model, device, {"variant": variant, "base_channels": base_channels, "depth": depth}, False


def show_image_triplet(image: Image.Image, mask, overlay) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Output saliency mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
    st.title("Salient Object Detection Demo")

    with st.sidebar:
        checkpoint_path = st.text_input("Checkpoint path", default_checkpoint_path())
        image_size = st.selectbox("Image size", [128, 224], index=0)
        variant = st.selectbox("Fallback model variant", ["baseline", "improved"], index=1)
        base_channels = st.selectbox("Fallback base channels", [8, 16, 32], index=1)
        depth = st.selectbox("Fallback depth", [3, 4], index=1)
        threshold_output = st.checkbox("Threshold mask", value=True)
        preset = st.selectbox(
            "Prediction preset",
            ["Balanced mask", "High precision", "Custom threshold"],
            index=0,
        )
        if preset == "Balanced mask":
            threshold = 0.55
        elif preset == "High precision":
            threshold = 0.94
        else:
            threshold = st.slider("Mask threshold", min_value=0.05, max_value=0.95, value=0.55, step=0.01)
        device_name = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    model, device, config, loaded = get_model(checkpoint_path, variant, base_channels, depth, device_name)
    if loaded:
        st.success(f"Loaded checkpoint on {device}: {checkpoint_path}")
    else:
        st.warning("Checkpoint not found. The app is showing an untrained model only for interface testing.")

    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if upload is None:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(upload).convert("RGB")
    mask, overlay, elapsed_ms = predict_image(
        model=model,
        image=image,
        device=device,
        image_size=int(image_size),
        threshold=float(threshold) if threshold_output else None,
    )
    st.metric("Inference time", f"{elapsed_ms:.1f} ms")
    show_image_triplet(image.resize((int(image_size), int(image_size))), mask, overlay)
    st.caption(f"Model config: {config}")


if __name__ == "__main__":
    main()
