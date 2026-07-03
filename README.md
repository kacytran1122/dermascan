# DermaScan — Skin Lesion Analysis

A deep-learning web app that classifies dermoscopic skin-lesion images as
**benign** or **malignant**, with a confidence score. The model is a fine-tuned
**EfficientNetV2L** (PyTorch) binary classifier trained on a melanoma dataset.

> ⚠️ **Not medical advice.** This is a research/educational demo, not a diagnostic
> device. Always consult a qualified dermatologist.

![Stack](https://img.shields.io/badge/model-EfficientNetV2L-6ea8ff)
![Backend](https://img.shields.io/badge/backend-FastAPI-2fd39b)
![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

## Features

- 🖼️ Drag-and-drop / browse image upload with live preview
- 🧠 Real-time inference via a FastAPI backend
- 📊 Animated confidence gauge and per-class probability bars
- 📱 Responsive, dark, glassmorphic UI
- 🔌 Simple JSON API (`/api/predict`, `/api/health`)

## Project structure

```
melanoma-model/
├── backend/
│   ├── app.py              # FastAPI server + inference
│   └── requirements.txt
├── frontend/
│   ├── index.html          # UI
│   ├── styles.css
│   └── app.js
├── notebooks/
│   ├── train.ipynb         # model training
│   └── test.ipynb          # single-image inference demo
├── model.pth               # trained weights (NOT in git — see below)
└── README.md
```

## Getting started

### 1. Add the model

The trained weights (`model.pth`, ~450 MB) are **not** committed to GitHub because
they exceed the 100 MB file limit. Place your `model.pth` in the project root, or
point the server at it:

```bash
export MODEL_PATH=/path/to/model.pth
```

### 2. Install dependencies

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
# from the backend/ directory
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## API

### `POST /api/predict`
Multipart form upload with an `file` image field.

```json
{
  "prediction": "Benign",
  "probability_malignant": 0.0123,
  "confidence": 0.9877,
  "logit": -4.39
}
```

### `GET /api/health`
```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

## Model details

- **Architecture:** EfficientNetV2L with a single-logit classifier head
- **Input:** 112×112 RGB, ImageNet normalization
- **Output:** one logit → `sigmoid` → P(malignant); threshold 0.5
- **Loss:** `BCEWithLogitsLoss`
- Reported validation accuracy ≈ **96%**

See `notebooks/train.ipynb` for the full training pipeline.
