# Deepfake Media Detection System

Detects whether a face image or video is **Real** or **Deepfake** using a
CNN (transfer learning) trained on the
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
dataset.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

Download **140k Real and Fake Faces** from Kaggle and place raw images into:
- `data/raw/real/` — real faces
- `data/raw/fake/` — GAN-generated faces

## Running the full pipeline

```bash
# 1. Preprocess
python -m src.preprocessing.face_extractor    # detect + crop faces
python -m src.preprocessing.dataset_builder   # stratified split -> data/processed/, data/splits/

# 2. Train (recommended: on Colab GPU — see Hardware note below)
python -m src.training.trainer

# 3. Evaluate on the held-out test set
python -m src.evaluation.evaluator            # -> reports/metrics.json, confusion_matrix.png, roc_curve.png

# 4. Run inference
python -m src.inference.image_predictor path/to/image.jpg
python -m src.inference.video_predictor path/to/video.mp4

# 5. Launch the app
streamlit run app/streamlit_app.py
```

## Project structure

```
deepfake-detection/
├── config.py                     # single source of truth for paths & hyperparameters
├── requirements.txt
├── data/
│   ├── raw/{real,fake}/          # your downloaded dataset goes here
│   ├── processed/{train,val,test}/{real,fake}/
│   └── splits/{train,val,test}.csv
├── src/
│   ├── preprocessing/
│   │   ├── face_extractor.py     # MTCNN face detection + cropping
│   │   └── dataset_builder.py    # stratified splitting + manifest CSVs
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset + DataLoader factory
│   │   └── augmentation.py       # Albumentations train/eval pipelines
│   ├── models/
│   │   └── xception_model.py     # timm-backed model factory + checkpoint I/O
│   ├── training/
│   │   └── trainer.py            # training loop, early stopping, checkpointing
│   ├── evaluation/
│   │   └── evaluator.py          # metrics, confusion matrix, ROC-AUC
│   ├── inference/
│   │   ├── image_predictor.py    # single-image prediction
│   │   └── video_predictor.py    # frame sampling + video-level prediction
│   └── utils/
│       └── logger.py
├── app/
│   └── streamlit_app.py          # deployment UI
├── checkpoints/                  # saved model weights
├── reports/                      # metrics.json, confusion_matrix.png, roc_curve.png (generated)
├── logs/
└── tests/
```

## Model choice

| Model | Why / why not |
|---|---|
| **Xception** | Depthwise-separable convs pretrained on ImageNet; this is the architecture the original FaceForensics++ paper used for deepfake detection, so it's a well-validated, defensible choice for an interview. Good accuracy-to-compute ratio. |
| **EfficientNet-B0** (default here) | Comparable accuracy, fewer params, faster to train — the default in this repo because the dev machine is CPU-only. Also a good fit for Streamlit Cloud's free tier. |
| ResNet50 | Solid baseline, well understood, slightly weaker on fine-grained artifact detection than Xception. |
| ViT-Base | Best ceiling accuracy with enough data, but needs more data/compute to avoid overfitting — riskier for a first pass on 140k images with limited GPU time. |

`config.py` lets you switch `TrainConfig.model_name` between these without touching code elsewhere.

## Hardware note (CPU-only training)

Confirmed dev machine: i5-1135G7, 8GB RAM, no CUDA GPU. `config.py` defaults
are tuned for this (`efficientnet_b0`, `batch_size=16`, `num_workers=2`). Use
this machine to build/test the pipeline end-to-end on a small subset of data.
For the **actual full training run**, use **Google Colab's free GPU tier** —
upload this repo, switch `TrainConfig.model_name` back to `"xception"` if you
want the paper-accurate architecture, bump `batch_size` to 32, and train
there. Download the resulting checkpoint back into `checkpoints/` for local
inference/Streamlit.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_dataset_builder.py` needs no ML libraries and runs standalone
(uses `opencv-python-headless`, `pandas`, `scikit-learn` only). The inference
tests (`test_image_predictor.py`, `test_video_predictor.py`) import the real
modules, so `torch` etc. from `requirements.txt` must be installed — they
bypass `__init__` (no checkpoint/MTCNN needed) and inject fake predictors to
test the aggregation/labeling *logic* in isolation.

---

## Project Status — core system complete

**Completed (full pipeline, end to end)**
- Project scaffolding (folder structure, config, logging)
- `src/preprocessing/face_extractor.py` — MTCNN-based face detection/cropping for images and video
- `src/preprocessing/dataset_builder.py` — stratified train/val/test split with CSV manifests
- `src/data/augmentation.py` — Albumentations pipelines (train: flips/rotation/color jitter/compression-blur-noise; eval: resize+normalize only)
- `src/data/dataset.py` — `DeepfakeFaceDataset` + `get_dataloaders()`
- `src/models/xception_model.py` — model factory (xception/efficientnet_b0/resnet50/vit_base) + checkpoint save/load
- `src/training/trainer.py` — training loop: `EarlyStopping`, `AdamW` + `ReduceLROnPlateau`, per-epoch history CSV, best-checkpoint saving
- `src/evaluation/evaluator.py` — accuracy/precision/recall/F1/ROC-AUC, confusion matrix + ROC curve plots, `reports/metrics.json`
- `src/inference/image_predictor.py` — single-image Real/Fake prediction with confidence
- `src/inference/video_predictor.py` — frame sampling + per-frame prediction + averaged video-level verdict
- `app/streamlit_app.py` — full upload UI (image tab, video tab, about/limitations tab)
- `tests/` — 13 unit tests: dataset-split logic (verified passing, 6/6), image/video prediction-aggregation logic (verified importable+syntax-correct; run locally with `torch` installed)

**Not yet done**
- Actual training run + real metrics/screenshots to paste into this README (do this after running the pipeline on real data)
- Optional extras: Grad-CAM visualization for interpretability, a FastAPI REST API alongside Streamlit, FaceForensics++ video-based v2 for face-swap/lip-sync deepfakes (a different artifact type than GAN still-image fakes)

**Recommended next step**
Get the dataset, run the pipeline end-to-end locally on a small subset (a few hundred images) to catch environment-specific bugs, then do the full training run on Colab. Once you have real metrics, paste them into this section and add 2-3 Streamlit screenshots — that's what turns this from "code" into a portfolio piece.
