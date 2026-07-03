# DermaScan

DermaScan looks at a photo of a skin lesion and gives a quick read on whether it
looks benign or malignant, along with a confidence score. It runs on an
EfficientNetV2L model we trained on a melanoma image dataset.

**Live app:** https://dermascan-b6rvmbxihuta9ylq72sw8g.streamlit.app

> DermaScan is a student project, not a medical device. It can't diagnose
> anything — if you're worried about a spot on your skin, see a dermatologist.

## The project

We built DermaScan for the **1 Idea 1 World** competition (2022–2023). Skin
cancer is very treatable when it's caught early, and most people don't have an
easy way to get a first opinion, so we trained a model that gives one in a few
seconds.

Team of four. Team lead: Kacy Tran.

## What it does

- Take a dermoscopic photo (JPG or PNG)
- Run it through the model
- Show a benign / malignant call, the probability for each class, and a
  confidence score

## Running it locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Get the model weights. `model.pth` is about 450 MB, so it isn't in the repo.
Two options:

- Download it from our Google Drive and drop `model.pth` in the project root:
  https://drive.google.com/drive/folders/1FoXg5obn5oLQG6qd_-EyBm8j0DYwyw2Y
- Or just run the app — if it doesn't find a local file, it pulls the weights
  from that same Drive folder on first launch.

Start the app:

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

## Layout

```
streamlit_app.py     the app and inference code
.streamlit/          theme
notebooks/           training and inference notebooks
requirements.txt
model.pth            weights (not committed — see above)
```
