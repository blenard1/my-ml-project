# Mentor Submission Checklist

Project: End-to-End Salient Object Detection  
Final model: `checkpoints/improved/best_model.pt`  
Dataset used: ECSSD, 1000 image/mask pairs

## Required Deliverables

| Assignment item | Status | File |
| --- | --- | --- |
| `data_loader.py` with dataset loading, preprocessing, augmentation | Done | `data_loader.py` |
| `sod_model.py` with full CNN model from scratch | Done | `sod_model.py` |
| `train.py` with training/validation loop and logging | Done | `train.py` |
| `evaluate.py` with metrics and visualization | Done | `evaluate.py` |
| Simple demo notebook or app | Done | `demo_notebook.ipynb`, `app.py` |
| Project report, 6-10 pages | Done | `artifacts/final_report.pdf` |
| Presentation slides, max 5 slides | Done | `artifacts/presentation_slides.pptx`, `artifacts/presentation_slides.pdf` |
| Trained model | Done | `checkpoints/improved/best_model.pt` |

## Final Results

| Mode | Threshold | IoU | Precision | Recall | F1 | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Improved balanced | 0.55 | 0.5364 | 0.6478 | 0.7711 | 0.6915 | 0.1935 |
| Improved high precision | 0.94 | 0.3426 | 0.8726 | 0.3622 | 0.5007 | 0.1935 |

## Files To Send Or Upload

Upload the full repository to GitHub, including:

- Source code: `data_loader.py`, `sod_model.py`, `train.py`, `evaluate.py`, `app.py`, `metrics.py`, `download_ecssd.py`.
- Demo: `demo_notebook.ipynb`.
- Trained model: `checkpoints/improved/best_model.pt`.
- Report: `artifacts/final_report.pdf`.
- Slides: `artifacts/presentation_slides.pptx` or `artifacts/presentation_slides.pdf`.
- Metrics/visual proof: `artifacts/evaluation/improved_balanced/` and `artifacts/evaluation/improved_precision085/`.

Do not upload the full `data/` folder unless your mentor specifically asks for it. The dataset can be downloaded with:

```powershell
python download_ecssd.py --data-dir data/ecssd
```

## Email Template

Use `artifacts/email_submission_template.txt`, paste the GitHub repository URL, and CC the AI capability lead as required by the assignment.
