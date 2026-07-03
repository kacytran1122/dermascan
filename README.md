# DermaScan — Skin Lesion Analysis

A Streamlit web app that classifies dermoscopic skin-lesion images as
**benign** or **malignant**, with a confidence score. The model is a fine-tuned
**EfficientNetV2L** (PyTorch) binary classifier trained on a melanoma dataset.

> ⚠️ **Not medical advice.** This is a research/educational demo, not a diagnostic
> device. Always consult a qualified dermatologist.

![Model](https://img.shields.io/badge/model-EfficientNetV2L-0f766e)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

## Features

- Image upload with preview (JPG / PNG)
- Real-time inference on the trained model
- Benign / malignant verdict with per-class probabilities and confidence
- Clean, clinical interface built with Streamlit

## Project structure

```
melanoma-model/
├── streamlit_app.py        # Streamlit UI + inference
├── .streamlit/
│   └── config.toml         # theme
├── requirements.txt
├── notebooks/
│   ├── train.ipynb         # model training
│   └── test.ipynb          # single-image inference demo
├── model.pth               # trained weights (not in git — see below)
└── README.md
```

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Provide the model weights

The trained weights (`model.pth`, ~450 MB) are **not** committed to GitHub
because they exceed the 100 MB file limit. You have two options:

- **Local file:** place `model.pth` in the project root (or set
  `export MODEL_PATH=/path/to/model.pth`).
- **Automatic download:** if no local file is found, the app downloads the
  weights from the shared Google Drive folder on first run (via `gdown`).

### 3. Run

```bash
streamlit run streamlit_app.py
```

Streamlit opens the app at **http://localhost:8501**.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `streamlit_app.py`.
3. The app downloads `model.pth` from Google Drive automatically on first run.

> Note: EfficientNetV2L is a large model. On the free Community Cloud tier
> (limited RAM), loading may be slow or memory-constrained; a machine with
> ≥ 2 GB RAM is recommended.

## Model details

- **Architecture:** EfficientNetV2L with a single-logit classifier head
- **Input:** 112×112 RGB, ImageNet normalization
- **Output:** one logit → `sigmoid` → P(malignant); threshold 0.5
- **Loss:** `BCEWithLogitsLoss`
- Reported validation accuracy ≈ **96%**

See `notebooks/train.ipynb` for the full training pipeline.
