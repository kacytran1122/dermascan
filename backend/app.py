"""
FastAPI backend for the melanoma (skin lesion) detection model.

Loads the trained EfficientNetV2L binary classifier and exposes a single
`/api/predict` endpoint that accepts an uploaded image and returns a
benign/malignant prediction with a confidence score.
"""

import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision.transforms import v2

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.pth"))
FRONTEND_DIR = BASE_DIR / "frontend"
IMG_SIZE = 112

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing used for validation/inference during training.
_transform = v2.Compose(
    [
        v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = None


def load_model():
    """Load the serialized model once at startup."""
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Place your trained 'model.pth' in the project root "
            "or set the MODEL_PATH environment variable."
        )
    # weights_only=False is required because the checkpoint stores the full
    # nn.Module object, not just a state_dict.
    loaded = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    loaded.eval()
    return loaded


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="Melanoma Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    global model
    try:
        model = load_model()
        print(f"Model loaded on {DEVICE}.")
    except Exception as exc:  # noqa: BLE001
        # Don't crash the server so the frontend still loads and can show a
        # helpful error; predictions will report the failure.
        print(f"WARNING: could not load model: {exc}")
        model = None


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(DEVICE)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure model.pth is present and restart the server.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        # Single-logit binary head -> sigmoid gives P(malignant).
        prob_malignant = torch.sigmoid(logit).item()

    is_malignant = prob_malignant >= 0.5
    confidence = prob_malignant if is_malignant else 1.0 - prob_malignant

    return {
        "prediction": "Malignant" if is_malignant else "Benign",
        "probability_malignant": round(prob_malignant, 4),
        "confidence": round(confidence, 4),
        "logit": round(logit.item(), 4),
    }


# --------------------------------------------------------------------------- #
# Serve the frontend (mounted last so /api/* takes precedence).
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.exists():

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
