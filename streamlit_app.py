"""
DermaScan — skin lesion analysis
A Streamlit interface for the EfficientNetV2L benign/malignant classifier.
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
IMG_SIZE = 112
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).resolve().parent
# Google Drive folder holding the trained weights (used when no local file).
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1FoXg5obn5oLQG6qd_-EyBm8j0DYwyw2Y"

st.set_page_config(
    page_title="DermaScan — Skin Lesion Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling — restrained, clinical
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 3rem; }
      #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

      .app-title { font-size: 1.9rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; margin: 0; }
      .app-sub   { color: #64748b; font-size: 0.98rem; margin-top: 0.2rem; }
      .rule      { height: 3px; width: 52px; background: #0f766e; border-radius: 3px; margin: 0.6rem 0 0.2rem; }

      .badge {
        display: inline-block; padding: 0.5rem 1.1rem; border-radius: 6px;
        font-weight: 700; font-size: 1.15rem; letter-spacing: 0.01em;
      }
      .badge-benign    { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
      .badge-malignant { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

      .section-label {
        text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem;
        color: #94a3b8; font-weight: 600; margin-bottom: 0.5rem;
      }
      .muted { color: #64748b; font-size: 0.85rem; }

      div[data-testid="stMetricValue"] { font-size: 1.7rem; }
      .stProgress > div > div > div { background-color: #0f766e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def resolve_model_path() -> Path:
    """Locate model.pth locally, or download it from Drive on first run."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    local = BASE_DIR / "model.pth"
    if local.exists():
        return local

    # Fall back to downloading the weights from the shared Drive folder.
    import gdown

    dest = BASE_DIR / "models"
    dest.mkdir(exist_ok=True)
    with st.spinner("Downloading model weights (first run only)…"):
        gdown.download_folder(DRIVE_FOLDER_URL, output=str(dest), quiet=True, use_cookies=False)

    weights = list(dest.glob("*.pth"))
    if not weights:
        raise FileNotFoundError("No .pth file found in the downloaded model folder.")
    return weights[0]


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import torch

    path = resolve_model_path()
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def predict(model, image: Image.Image):
    import torch
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logit = model(tensor)
        prob_malignant = torch.sigmoid(logit).item()

    return {
        "prob_malignant": prob_malignant,
        "prob_benign": 1.0 - prob_malignant,
        "label": "Malignant" if prob_malignant >= 0.5 else "Benign",
        "logit": logit.item(),
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### About")
    st.write(
        "DermaScan analyzes a dermoscopic image of a skin lesion and estimates "
        "whether it appears **benign** or **malignant**."
    )

    st.markdown("### Model")
    st.markdown(
        "- **Architecture:** EfficientNetV2L\n"
        "- **Input:** 112×112 RGB\n"
        "- **Output:** single logit → P(malignant)\n"
        "- **Validation accuracy:** ≈ 96%"
    )

    st.markdown("### How to use")
    st.markdown(
        "1. Upload a lesion image (JPG or PNG)\n"
        "2. Select **Analyze**\n"
        "3. Review the prediction and confidence"
    )

    st.divider()
    st.caption(
        "Research and educational use only. Not a medical device and not a "
        "substitute for professional diagnosis."
    )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="app-title">DermaScan</p>', unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Skin lesion classification — benign vs. malignant</p>',
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-label">Input image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a dermoscopic image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

    analyze = st.button(
        "Analyze", type="primary", use_container_width=True, disabled=uploaded is None
    )

with right:
    st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)

    if analyze and image is not None:
        try:
            model = load_model()
            result = predict(model, image)

            badge_class = "badge-malignant" if result["label"] == "Malignant" else "badge-benign"
            st.markdown(
                f'<span class="badge {badge_class}">{result["label"]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")

            c1, c2 = st.columns(2)
            c1.metric("P(malignant)", f"{result['prob_malignant'] * 100:.1f}%")
            confidence = max(result["prob_malignant"], result["prob_benign"])
            c2.metric("Confidence", f"{confidence * 100:.1f}%")

            st.markdown('<p class="muted">Benign</p>', unsafe_allow_html=True)
            st.progress(result["prob_benign"])
            st.markdown('<p class="muted">Malignant</p>', unsafe_allow_html=True)
            st.progress(result["prob_malignant"])

            st.caption(f"Raw model output (logit): {result['logit']:.3f}")

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run the model: {exc}")
    else:
        st.info("Upload an image and select **Analyze** to see the prediction.")

# --------------------------------------------------------------------------- #
# Footer / disclaimer
# --------------------------------------------------------------------------- #
st.write("")
st.divider()
st.caption(
    "⚕️ **Disclaimer** — DermaScan is a research and educational demonstration. "
    "It is not a diagnostic device and must not be used to make medical decisions. "
    "Consult a qualified dermatologist regarding any concern about your skin."
)
