"""
DermaScan — skin lesion analysis
Streamlit interface for the EfficientNetV2L benign/malignant classifier.
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
POSTER_PATH = BASE_DIR / "MELANOMA CANCER PREDICTION POSTER.jpg"
# Google Drive folder holding the trained weights (used when no local file).
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1FoXg5obn5oLQG6qd_-EyBm8j0DYwyw2Y"

st.set_page_config(
    page_title="DermaScan — Melanoma Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Styling — clean business / SaaS
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      :root { --blue: #2563eb; --blue-dark: #1d4ed8; --ink: #0f172a; --slate: #475569;
              --line: #e2e8f0; --bg: #f8fafc; }

      .stApp { background: var(--bg); color: var(--ink);
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .block-container { max-width: 1120px; padding-top: 1.2rem; padding-bottom: 4rem; }
      #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

      /* Navbar */
      .nav { display: flex; align-items: center; justify-content: space-between;
        padding: 0.9rem 0 1.1rem; border-bottom: 1px solid var(--line); margin-bottom: 2.4rem; }
      .nav-brand { display: flex; align-items: center; gap: 0.6rem; font-weight: 700; font-size: 1.12rem; color: var(--ink); }
      .nav-mark { width: 26px; height: 26px; border-radius: 7px; background: linear-gradient(135deg, #3b82f6, #2563eb);
        display: grid; place-items: center; color: #fff; font-size: 0.9rem; box-shadow: 0 2px 6px rgba(37,99,235,0.35); }
      .nav-links { display: flex; gap: 1.6rem; font-size: 0.9rem; color: var(--slate); }
      .nav-links span { font-weight: 500; }
      .nav-cta { display: inline-flex; align-items: center; gap: 7px; font-size: 0.8rem; font-weight: 600; color: #047857;
        background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 999px; padding: 6px 13px; }
      .nav-cta .dot { width: 7px; height: 7px; border-radius: 50%; background: #10b981; }

      /* Hero */
      .eyebrow { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--blue); }
      .h1 { font-size: clamp(2rem, 4.6vw, 3rem); font-weight: 800; letter-spacing: -0.025em; line-height: 1.08;
        color: var(--ink); margin: 0.6rem 0 0; }
      .sub { color: var(--slate); font-size: 1.05rem; line-height: 1.6; max-width: 42rem; margin-top: 0.9rem; }

      /* KPI strip */
      .kpi { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04); height: 100%; }
      .kpi-num { font-size: 1.5rem; font-weight: 800; color: var(--ink); letter-spacing: -0.01em; }
      .kpi-label { font-size: 0.82rem; color: var(--slate); margin-top: 0.15rem; }

      /* Section headers */
      .sec-eyebrow { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--blue); }
      .sec-title { font-size: 1.5rem; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; margin: 0.25rem 0 0.2rem; }
      .sec-sub { color: var(--slate); font-size: 0.96rem; margin-bottom: 0.4rem; }
      .card-head { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #94a3b8; }
      .card-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin-top: 0.15rem; }

      /* Cards (bordered containers) */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border: 1px solid var(--line) !important; border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05), 0 1px 2px rgba(15,23,42,0.03);
      }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"] { background: var(--bg); border: 1.4px dashed #cbd5e1; border-radius: 11px; }
      [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--blue); background: #eff6ff; }

      /* Primary button — solid blue, business rectangle */
      .stButton > button[kind="primary"] {
        background: var(--blue); color: #fff; border: none; border-radius: 9px; font-weight: 600;
        padding: 0.65rem 1rem; box-shadow: 0 1px 2px rgba(37,99,235,0.35); transition: background 0.15s, transform 0.1s; }
      .stButton > button[kind="primary"]:hover:not(:disabled) { background: var(--blue-dark); }
      .stButton > button[kind="primary"]:active:not(:disabled) { transform: translateY(1px); }
      .stButton > button[kind="primary"]:disabled { background: #cbd5e1; color: #fff; box-shadow: none; }

      /* Verdict badge */
      .badge { display: inline-flex; align-items: center; gap: 8px; padding: 0.5rem 1.1rem; border-radius: 8px;
        font-weight: 700; font-size: 1.1rem; }
      .badge .dot { width: 9px; height: 9px; border-radius: 50%; }
      .badge-benign { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
      .badge-benign .dot { background: #10b981; }
      .badge-malignant { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
      .badge-malignant .dot { background: #ef4444; }

      /* Metrics */
      [data-testid="stMetricValue"] { font-size: 1.6rem; color: var(--ink); font-weight: 800; }
      [data-testid="stMetricLabel"] p { color: var(--slate) !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.05em; }
      .stProgress > div > div > div > div { background-image: linear-gradient(90deg, #3b82f6, #2563eb); }
      .stProgress > div > div > div { background-color: #e2e8f0; }
      .muted { color: var(--slate); font-size: 0.82rem; margin-bottom: 0.25rem; }

      /* Feature / roadmap cards */
      .feat { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 1.3rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04); height: 100%; }
      .feat-ico { width: 38px; height: 38px; border-radius: 9px; background: #eff6ff; color: var(--blue);
        display: grid; place-items: center; font-weight: 700; margin-bottom: 0.8rem; }
      .feat-title { font-weight: 700; color: var(--ink); font-size: 1.02rem; margin-bottom: 0.35rem; }
      .feat-desc { color: var(--slate); font-size: 0.9rem; line-height: 1.55; }
      .feat-tag { display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
        color: #7c3aed; background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 6px; padding: 2px 8px; margin-bottom: 0.7rem; }

      .credit-pill { display: inline-block; padding: 7px 15px; border-radius: 999px; font-size: 0.85rem;
        color: var(--ink); background: #fff; border: 1px solid var(--line); box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
      .medal { color: #b45309; font-weight: 700; }
      hr { border-color: var(--line); }
      .disc { color: #94a3b8; font-size: 0.82rem; line-height: 1.55; }
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
# Navbar
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="nav">
      <div class="nav-brand"><span class="nav-mark">◆</span>DermaScan</div>
      <div class="nav-links">
        <span>Overview</span><span>How it works</span><span>Roadmap</span><span>About</span>
      </div>
      <div class="nav-cta"><span class="dot"></span>Model online</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown('<p class="eyebrow">Melanoma screening</p>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="h1">Early skin-cancer screening,<br>powered by deep learning</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub">Melanoma is a small share of skin cancers but causes most skin-cancer deaths — '
    "and it is highly treatable when caught early. DermaScan gives a fast first-pass read on a lesion "
    "image: benign or malignant, with a confidence score.</p>",
    unsafe_allow_html=True,
)
st.write("")

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown('<div class="kpi"><div class="kpi-num">≈ 96%</div>'
                '<div class="kpi-label">Validation accuracy</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><div class="kpi-num">EfficientNetV2L</div>'
                '<div class="kpi-label">Model architecture</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><div class="kpi-num">&lt; 1s</div>'
                '<div class="kpi-label">Time per prediction (GPU)</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
st.markdown('<p class="sec-eyebrow">Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sec-title">Screen a lesion image</p>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<p class="card-head">Step 1 — Input</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Upload an image</p>', unsafe_allow_html=True)
        st.write("")
        uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        image = None
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)

        st.write("")
        analyze = st.button(
            "Analyze lesion", type="primary", use_container_width=True, disabled=uploaded is None
        )

with right:
    with st.container(border=True):
        st.markdown('<p class="card-head">Step 2 — Result</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Model assessment</p>', unsafe_allow_html=True)
        st.write("")

        if analyze and image is not None:
            try:
                model = load_model()
                result = predict(model, image)
                is_mal = result["label"] == "Malignant"
                cls = "badge-malignant" if is_mal else "badge-benign"

                st.markdown(
                    f'<span class="badge {cls}"><span class="dot"></span>{result["label"]}</span>',
                    unsafe_allow_html=True,
                )
                st.write("")

                c1, c2 = st.columns(2)
                c1.metric("P(malignant)", f"{result['prob_malignant'] * 100:.1f}%")
                confidence = max(result["prob_malignant"], result["prob_benign"])
                c2.metric("Confidence", f"{confidence * 100:.1f}%")

                st.write("")
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
            st.markdown(
                '<p style="color:#94a3b8;padding:2rem 0;">'
                "Upload an image and select <b>Analyze lesion</b> to see the prediction.</p>",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# How it works
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.markdown('<p class="sec-eyebrow">How it works</p>', unsafe_allow_html=True)
st.markdown('<p class="sec-title">From photo to prediction</p>', unsafe_allow_html=True)
st.write("")

steps = [
    ("1", "Upload", "Add a dermoscopic photo of the lesion. No manual scoring — the model does the reading."),
    ("2", "Analyze", "The image is resized to 112×112 and passed through an EfficientNetV2L network."),
    ("3", "Read the result", "You get a benign / malignant call, per-class probabilities, and a confidence score."),
]
cols = st.columns(3, gap="medium")
for col, (n, title, desc) in zip(cols, steps):
    col.markdown(
        f'<div class="feat"><div class="feat-ico">{n}</div>'
        f'<div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Roadmap / future directions
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.markdown('<p class="sec-eyebrow">Roadmap</p>', unsafe_allow_html=True)
st.markdown('<p class="sec-title">Where DermaScan goes next</p>', unsafe_allow_html=True)
st.write("")

roadmap = [
    ("Broader, more diverse data",
     "Grow the training set and balance it across skin tones and lesion types so the model stays "
     "accurate for everyone, not just the cases it saw most."),
    ("Clinical-grade interface",
     "Add heatmaps that show which region drove a prediction, lesion tracking over time, and "
     "exportable reports a clinician can actually use."),
    ("Mobile app",
     "Bring screening to the phone camera, with on-device inference, so a first-pass check works "
     "anywhere — including places far from a dermatologist."),
]
rcols = st.columns(3, gap="medium")
for col, (title, desc) in zip(rcols, roadmap):
    col.markdown(
        f'<div class="feat"><span class="feat-tag">Planned</span>'
        f'<div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.divider()
about_l, about_r = st.columns([1, 1], gap="large")
with about_l:
    st.markdown('<p class="sec-eyebrow">About the project</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Built by students for early detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-sub">DermaScan was built for the <b>1 Idea 1 World</b> competition '
        '(<span class="medal">Gold Medal</span>, 2022–2023). Melanoma is one of the most treatable '
        "cancers when caught early, and most people have no fast way to get a first opinion — so we "
        "trained a model that provides one.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="credit-pill">1 Idea 1 World · <span class="medal">Gold Medal</span> · 2022–2023 '
        "&nbsp;·&nbsp; Team of 4 &nbsp;·&nbsp; Team lead: Kacy Tran</span>",
        unsafe_allow_html=True,
    )
with about_r:
    if POSTER_PATH.exists():
        st.image(str(POSTER_PATH), use_container_width=True, caption="Competition research poster")

# --------------------------------------------------------------------------- #
# Disclaimer
# --------------------------------------------------------------------------- #
st.write("")
st.divider()
st.markdown(
    '<p class="disc"><b>Disclaimer.</b> DermaScan is a research and educational project. '
    "It is not a medical device and is not a substitute for professional diagnosis. "
    "See a qualified dermatologist about any concern with your skin.</p>",
    unsafe_allow_html=True,
)
