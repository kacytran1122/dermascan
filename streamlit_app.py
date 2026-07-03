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
# Google Drive folder holding the trained weights (used when no local file).
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1FoXg5obn5oLQG6qd_-EyBm8j0DYwyw2Y"

st.set_page_config(
    page_title="DermaScan — Skin Lesion Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      :root { --sky: #38bdf8; --sky-soft: #7dd3fc; }

      .stApp {
        background:
          radial-gradient(1100px 620px at 82% -8%, rgba(56,189,248,0.16), transparent 60%),
          radial-gradient(900px 600px at -5% 108%, rgba(45,120,220,0.12), transparent 55%),
          #0a0a0a;
        color: #ededed;
        font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
      }
      .block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 4rem; }
      #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

      /* Top bar */
      .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 3.2rem; }
      .wordmark { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.16em; }
      .tagpill {
        font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--sky-soft);
        border: 1px solid rgba(56,189,248,0.35); border-radius: 999px; padding: 5px 13px;
        background: rgba(56,189,248,0.07);
      }

      /* Hero */
      .eyebrow { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--sky); }
      .hero-h1 {
        font-size: clamp(2.4rem, 6vw, 4rem); font-weight: 800; line-height: 1.03;
        letter-spacing: -0.02em; margin: 0.7rem 0 0; color: #ffffff;
      }
      .hero-h1 .accent { color: var(--sky); }
      .hero-sub { color: rgba(237,237,237,0.62); font-size: 1.08rem; line-height: 1.6; max-width: 40rem; margin-top: 1.1rem; }

      .section-eyebrow {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.13em; text-transform: uppercase;
        color: rgba(237,237,237,0.42); margin-bottom: 0.7rem;
      }
      .card-head { font-size: 1.05rem; font-weight: 700; color: #fff; }

      /* Glass cards via bordered container */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(10px);
      }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02); border: 1.4px dashed rgba(255,255,255,0.16);
        border-radius: 14px; transition: border-color 0.2s, background 0.2s;
      }
      [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--sky); background: rgba(56,189,248,0.05); }
      [data-testid="stFileUploaderDropzone"] * { color: rgba(237,237,237,0.72) !important; }
      [data-testid="stBaseButton-secondary"] {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.18);
        color: #fff; border-radius: 999px;
      }

      /* Primary button — white pill, dark text (Fitza CTA) */
      .stButton > button[kind="primary"] {
        background: #ffffff; color: #0a0a0a; border: none; border-radius: 999px;
        font-weight: 700; padding: 0.7rem 1rem; transition: transform 0.12s, background 0.2s, box-shadow 0.2s;
        box-shadow: 0 10px 30px -12px rgba(56,189,248,0.55);
      }
      .stButton > button[kind="primary"]:hover:not(:disabled) { background: #e0f2fe; transform: translateY(-1px); }
      .stButton > button[kind="primary"]:active:not(:disabled) { transform: scale(0.98); }
      .stButton > button[kind="primary"]:disabled { background: rgba(255,255,255,0.14); color: rgba(255,255,255,0.4); box-shadow: none; }

      /* Verdict badge */
      .badge {
        display: inline-flex; align-items: center; gap: 8px; padding: 0.55rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 1.2rem; letter-spacing: 0.01em;
      }
      .badge .dot { width: 9px; height: 9px; border-radius: 50%; }
      .badge-benign { background: rgba(56,189,248,0.12); color: var(--sky-soft); border: 1px solid rgba(56,189,248,0.4); }
      .badge-benign .dot { background: var(--sky); box-shadow: 0 0 12px 2px rgba(56,189,248,0.7); }
      .badge-malignant { background: rgba(244,63,94,0.12); color: #fda4af; border: 1px solid rgba(244,63,94,0.42); }
      .badge-malignant .dot { background: #f43f5e; box-shadow: 0 0 12px 2px rgba(244,63,94,0.7); }

      /* Metrics */
      [data-testid="stMetricValue"] { font-size: 1.7rem; color: #fff; font-weight: 700; }
      [data-testid="stMetricLabel"] p { color: rgba(237,237,237,0.5) !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.08em; }
      .stProgress > div > div > div > div { background-image: linear-gradient(90deg, #2563eb, var(--sky)); }
      .stProgress > div > div > div { background-color: rgba(255,255,255,0.07); }
      .muted { color: rgba(237,237,237,0.55); font-size: 0.82rem; margin-bottom: 0.25rem; }

      /* Timeline steps */
      .step { display: flex; gap: 1rem; padding: 0.55rem 0; }
      .step-num {
        flex: 0 0 auto; width: 2.4rem; height: 2.4rem; border-radius: 50%;
        display: grid; place-items: center; font-weight: 700; font-size: 0.85rem; color: var(--sky-soft);
        border: 1px solid rgba(56,189,248,0.5); background: rgba(10,10,10,0.6);
      }
      .step-title { font-weight: 600; color: #fff; margin: 0.2rem 0 0.15rem; }
      .step-desc { color: rgba(237,237,237,0.55); font-size: 0.9rem; line-height: 1.5; }

      .about-h { font-size: clamp(1.8rem, 4vw, 2.6rem); font-weight: 800; color: #fff; letter-spacing: -0.02em; }
      .about-p { color: rgba(237,237,237,0.62); line-height: 1.65; font-size: 1rem; max-width: 42rem; }
      .credit-pill {
        display: inline-block; margin-top: 0.4rem; padding: 6px 14px; border-radius: 999px; font-size: 0.82rem;
        color: rgba(237,237,237,0.8); background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
      }
      hr { border-color: rgba(255,255,255,0.08); }
      .disc { color: rgba(237,237,237,0.42); font-size: 0.82rem; line-height: 1.55; }
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
# Top bar + hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="topbar">
      <span class="wordmark">DERMASCAN</span>
      <span class="tagpill">EfficientNetV2 · PyTorch</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="eyebrow">AI-assisted dermatology</p>
    <h1 class="hero-h1">Know your skin,<br><span class="accent">before it speaks.</span></h1>
    <p class="hero-sub">
      Upload a dermoscopic image and DermaScan estimates whether a lesion looks
      benign or malignant — in seconds, with a confidence score you can read at a glance.
    </p>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.write("")

# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<p class="section-eyebrow">01 — Input</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-head">Upload a lesion image</p>', unsafe_allow_html=True)
        st.write("")
        uploaded = st.file_uploader(
            "Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

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
        st.markdown('<p class="section-eyebrow">02 — Result</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-head">Model assessment</p>', unsafe_allow_html=True)
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
                '<p style="color:rgba(237,237,237,0.5);padding:2.2rem 0;">'
                "Upload an image and select <b>Analyze lesion</b> to see the prediction.</p>",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# How it works
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.markdown('<p class="eyebrow">How it works</p>', unsafe_allow_html=True)
st.markdown(
    '<h2 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0.3rem 0 1.4rem;letter-spacing:-0.02em;">'
    "Three steps, no guesswork</h2>",
    unsafe_allow_html=True,
)

steps = [
    ("Upload", "Drop a dermoscopic photo of the lesion. Everything runs on the model — no manual scoring."),
    ("Analyze", "An EfficientNetV2L network trained on thousands of images reads the lesion at 112×112."),
    ("Read the result", "Get a benign / malignant call, per-class probabilities, and a confidence score."),
]
for i, (title, desc) in enumerate(steps, 1):
    st.markdown(
        f"""
        <div class="step">
          <div class="step-num">{i:02d}</div>
          <div>
            <p class="step-title">{title}</p>
            <p class="step-desc">{desc}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.divider()
st.markdown('<p class="eyebrow">About the project</p>', unsafe_allow_html=True)
st.markdown('<p class="about-h">Built by students, for early detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="about-p">DermaScan started as a student project for the <b>1 Idea 1 World</b> '
    "competition (2022–2023). Skin cancer is one of the most treatable cancers when caught early — "
    "so we trained a model that puts a fast, first-pass screening tool in anyone's hands.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="credit-pill">1 Idea 1 World · 2022–2023 &nbsp;·&nbsp; Team of 4 &nbsp;·&nbsp; '
    "Team lead: Kacy Tran</span>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Disclaimer
# --------------------------------------------------------------------------- #
st.write("")
st.write("")
st.markdown(
    '<p class="disc"><b>Disclaimer.</b> DermaScan is a research and educational project. '
    "It is not a medical device and is not a substitute for professional diagnosis. "
    "See a qualified dermatologist about any concern with your skin.</p>",
    unsafe_allow_html=True,
)
