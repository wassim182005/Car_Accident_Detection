# Accident Detection from CCTV Footage

A deep learning system that classifies CCTV frames as **accident** or **non-accident**, built with transfer learning and deployed through a Streamlit web interface.

## Overview

Road accidents often go unnoticed in CCTV footage until reviewed manually. This project automates that detection using a fine-tuned convolutional neural network, enabling faster identification of accident events from surveillance frames.

## Tech Stack

`Python` `TensorFlow / Keras` `MobileNetV3Large` `Streamlit` `NumPy` `PIL`

## Approach

- **Data preparation:** cleaned a Kaggle CCTV image dataset (corrupt file removal, duplicate detection, class balance verification)
- **Modeling:** transfer learning on MobileNetV3Large, with a custom classification head
- **Training strategy:** two-phase training — frozen backbone, then fine-tuning of the upper layers at a reduced learning rate
- **Class imbalance:** handled via computed class weights
- **Deployment:** Streamlit web app for image upload and real-time prediction

## Results

| Metric | Validation |
|---|---|
| Accuracy | 89.01% |
| Precision | 84.31% |
| Recall | 95.56% |

## Demo

The Streamlit interface lets users upload a CCTV frame and instantly view the model's prediction — accident or no accident — along with a confidence score.

## Project Structure

```
.
├── APP/
│   └── app.py            # Streamlit web app
├── notebook/
│   ├── notebook.ipynb               # Data cleaning, training, fine-tuning
│   └── Rawane_wassim.keras          # Trained model
├── .gitignore
├── requirements.txt
└── README.md

```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```
