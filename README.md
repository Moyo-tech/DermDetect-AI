# DermDetect-AI

DermDetect-AI is an **experimental, educational ML prototype** for classifying dermatoscopic skin-lesion images. It pairs a DenseNet121-based model with a Streamlit interface that displays ranked dataset labels for an image. It is **not a medical device or diagnostic service**. Its outputs must not be used to decide whether a lesion is cancerous, whether to seek care, or what treatment to pursue.

[Project walkthrough](https://drive.google.com/file/d/10J2jSXMi_5FvK3Yl3xfoDVm3kq8uSZrW/view?usp=sharing) · [Streamlit demo](https://dermdetect-ai.streamlit.app/) (availability may vary)

## What is in this repository

- `Notebooks/` documents data preparation, training, and experimentation.
- `model/optimisedmodel121.h5` is the saved model used by the demo.
- `app.py` loads that model, preprocesses uploaded or example images, and displays the top predicted class labels in Streamlit.
- `assets/modelclasses.txt` maps model outputs to the seven lesion classes shown in the app.

The project documentation describes work with HAM10000 and ISIC 2018 training images. The classes are AKIEC, BCC, BKL, DF, MEL, NV, and VASC. These dataset labels and model scores are not clinical diagnoses or calibrated probabilities.

## Model and interface

DenseNet121 was used as a transfer-learning backbone for image classification. The Streamlit app resizes an image to 224 × 224, scales pixel values, runs inference, and shows four ranked outputs. Its image relevance check compares an upload to a reference image using structural similarity. That check is a prototype UI filter, **not** a reliable way to tell whether an image depicts a skin lesion.

The app also accepts optional demo metadata such as age and anatomical site for a downloadable CSV. Do not enter real patient information or upload identifiable medical images to a public demo. The repository does not document clinical validation, prospective testing, subgroup performance, calibration, or performance on ordinary phone photographs. Dataset overlap, selection bias, and image distribution shift may affect results.

## Run locally

```bash
git clone https://github.com/Moyo-tech/DermDetect-AI.git
cd DermDetect-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The checked-in model file is required for inference. The pinned dependencies are from the original experiment and may require a compatible Python environment. The local setup was not rerun during this documentation update.

## Status

This project demonstrates an ML workflow and interface, not a validated healthcare product. Its earlier README described potential diagnostic impact that the repository does not establish. The current documentation intentionally makes no claim about diagnostic accuracy, clinical benefit, or generalization beyond the documented datasets.
