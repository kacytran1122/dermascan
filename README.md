# DermaScan

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

DermaScan looks at a photo of a skin lesion and gives a quick read on whether it
looks benign or malignant, along with a confidence score. It runs on an
EfficientNetV2L model we trained on a melanoma image dataset.

**Live app:** https://dermascan-b6rvmbxihuta9ylq72sw8g.streamlit.app

> DermaScan is a student project, not a medical device. It can't diagnose
> anything. If you're worried about a spot on your skin, see a dermatologist.

## The project

We built DermaScan for the **1 Idea 1 World** competition, where it won a
**Gold Medal** (2022–2023). Melanoma is only a small share of skin cancers but
causes most skin-cancer deaths, and it's very treatable when it's caught early.
Most people don't have an easy way to get a first opinion, so we trained a model
that gives one in a few seconds.

Team of four. Team lead: Kacy Tran. The research poster is in this repo:
[`MELANOMA CANCER PREDICTION POSTER.jpg`](MELANOMA%20CANCER%20PREDICTION%20POSTER.jpg)
and [`MELANOMA CANCER PREDICTION.pdf`](MELANOMA%20CANCER%20PREDICTION.pdf).

## What it does

- Take a dermoscopic photo (JPG or PNG)
- Run it through the model
- Show a benign / malignant call, the probability for each class, and a
  confidence score

## Running it locally

You'll need Python 3.9 or newer. The steps are the same on every platform.
Only the virtual-environment activation command differs.

**1. Get the code**

```bash
git clone https://github.com/kacytran1122/dermascan.git
cd dermascan
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Get the model weights**

`model.pth` is about 450 MB, so it isn't in the repo. Two options:

- Download it from our Google Drive and drop `model.pth` in the project root:
  https://drive.google.com/drive/folders/1FoXg5obn5oLQG6qd_-EyBm8j0DYwyw2Y
- Or just run the app. If it doesn't find a local file, it pulls the weights
  from that same Drive folder on first launch.

**5. Run the app**

```bash
streamlit run streamlit_app.py
```

It opens at http://localhost:8501.

## How the model works

- EfficientNetV2L with the classifier swapped for a single output
- Input images are resized to 112×112 and normalized with ImageNet stats
- One logit, run through a sigmoid, gives the probability of malignant
- Trained with `BCEWithLogitsLoss`; validation accuracy landed around 96%

The full training run is in `notebooks/train.ipynb`, and
`notebooks/test.ipynb` shows inference on a single image.

## Future directions

**Broader, more diverse data.** The biggest lever for accuracy is the dataset.
Right now the model learns from a limited set of images, which means it performs
best on the kinds of cases and skin tones it saw most often during training. The
plan is to gather more data and deliberately balance it across skin tones,
lesion types, and imaging conditions, so the model generalizes to real patients
instead of overfitting to a narrow slice of them. More and cleaner data is the
most direct path to a model people can actually trust.

**A clinical-grade interface.** A yes/no answer isn't enough for a tool meant to
support real decisions. The next step is to make the output explainable and
practical: heatmaps that highlight the region of the lesion that drove the
prediction, the ability to track a mole across multiple visits to see whether
it's changing, and clean, exportable reports a clinician or patient can keep on
file. The goal is to move DermaScan from a demo that returns a label to a tool
that shows its reasoning and fits into how people already keep track of their
skin.

**A mobile app.** Screening is most useful when it's available at the moment
someone notices a spot, not only when they're at a computer. A native mobile app
would let people point their phone camera at a lesion and get an instant
first-pass read, ideally with the model running on-device so it works offline and
keeps images private. That matters most for people who live far from a
dermatologist or can't easily get an appointment. Putting a quick, free check in
everyone's pocket is the whole point of the project.

## Layout

```
streamlit_app.py     the app and inference code
.streamlit/          theme
notebooks/           training and inference notebooks
requirements.txt
model.pth            weights (not committed, see above)
```
