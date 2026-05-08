# Salient Object Detection From Scratch

This repository implements the coursework project shown in the assignment screenshots: a full salient object detection pipeline using a CNN encoder-decoder trained from scratch. The model takes an RGB image and predicts a one-channel saliency mask, then visualizes the mask and an overlay on the original image.

The default dataset target is ECSSD because it is public, compact, and includes paired images and pixel masks. The official ECSSD page describes the dataset as 1000 complex natural images with helper-produced ground-truth masks: <https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html>.

## Project Files

- `data_loader.py`: paired image/mask loading, preprocessing, augmentation, splits, and a synthetic smoke dataset.
- `sod_model.py`: CNN encoder-decoder architecture implemented from scratch.
- `metrics.py`: BCE + IoU loss and IoU/precision/recall/F1/MAE metrics.
- `train.py`: training loop with forward/backward passes, validation, CSV logging, early stopping, and checkpoint resume.
- `evaluate.py`: test evaluation, metric export, prediction grids, and single-image inference helpers.
- `app.py`: Streamlit demo for uploading an image and viewing mask, overlay, and inference time.
- `demo_notebook.ipynb`: notebook workflow for loading a trained checkpoint and running inference.
- `download_ecssd.py`: ECSSD downloader using the official CUHK links.
- `checkpoints/improved/best_model.pt`: final trained model checkpoint.
- `artifacts/final_report.md`: report source.
- `artifacts/final_report.pdf`: final PDF report.
- `artifacts/presentation_slides.pptx`: five-slide editable presentation.
- `artifacts/presentation_slides.pdf`: PDF export of the same slides.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The code avoids pretrained models and only uses PyTorch modules defined in this repository.

## Download ECSSD

```powershell
python download_ecssd.py --data-dir data/ecssd
```

The expected layout after extraction is flexible. The loader pairs images and masks by filename stem, so both of these layouts work:

```text
data/ecssd/images/*.jpg
data/ecssd/ground_truth_mask/*.png
```

or any nested folders with matching image and mask stems.

## Train

Baseline:

```powershell
python train.py --data-dir data/ecssd --output-dir checkpoints/baseline --epochs 15 --image-size 128 --batch-size 4 --base-channels 16 --depth 4 --variant baseline --patience 15 --sample-every 5
```

Improved run with U-Net-style skip connections, batch normalization, and dropout:

```powershell
python train.py --data-dir data/ecssd --output-dir checkpoints/improved --epochs 15 --image-size 128 --batch-size 4 --base-channels 16 --depth 4 --variant improved --patience 15 --sample-every 5
```

Resume an interrupted run:

```powershell
python train.py --data-dir data/ecssd --output-dir checkpoints/baseline --resume
```

For a quick pipeline check without downloading data:

```powershell
python train.py --output-dir checkpoints/smoke --epochs 1 --synthetic-samples 24 --batch-size 4 --image-size 64 --base-channels 8
```

## Evaluate

```powershell
python evaluate.py --data-dir data/ecssd --checkpoint checkpoints/improved/best_model.pt --output-dir artifacts/evaluation/improved_balanced --threshold 0.55
```

This writes `metrics.json`, `metrics.csv`, and example prediction grids with input image, ground truth, predicted mask, and overlay.

## Run Demo

```powershell
streamlit run app.py
```

The app loads `checkpoints/improved/best_model.pt` by default. Use `Balanced mask` for the best visual segmentation and `High precision` for the 0.85+ precision operating point.

## Final Experiment Table

The final report uses the metrics exported by `evaluate.py`.

| Run | Change | IoU | Precision | Recall | F1 | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | Threshold 0.50 | 0.4651 | 0.6702 | 0.6175 | 0.6278 | 0.2158 |
| High precision | Same checkpoint, threshold 0.85 | 0.1231 | 0.8656 | 0.1255 | 0.2080 | 0.2158 |
| Improved balanced | U-Net skips + BatchNorm/Dropout, threshold 0.55 | 0.5364 | 0.6478 | 0.7711 | 0.6915 | 0.1935 |
| Improved high precision | U-Net skips + BatchNorm/Dropout, threshold 0.94 | 0.3426 | 0.8726 | 0.3622 | 0.5007 | 0.1935 |

To reproduce the final improved precision/recall tradeoff:

```powershell
python scripts/threshold_sweep.py --data-dir data/ecssd --checkpoint checkpoints/improved/best_model.pt --output artifacts/evaluation/improved_threshold_sweep.csv
```

For the final improved high-precision result:

```powershell
python evaluate.py --data-dir data/ecssd --checkpoint checkpoints/improved/best_model.pt --output-dir artifacts/evaluation/improved_precision085 --threshold 0.94
```
