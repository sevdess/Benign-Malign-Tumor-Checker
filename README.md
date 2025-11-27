# Benign-Malign-Tumor-Checker
Canine tumors data-sets using with MONAI 
MONAI Pathology Tumor Detection System

Language: English | Türkçe için aşağıya bakın

A simple Streamlit-based app that uses MONAI and PyTorch to perform tumor detection on histopathology images. This repo includes demo UIs and a pre-trained model reference packaged under pathology_tumor_detection/.

Features

Automated tumor detection on histopathology images
Confidence scoring for detections
Simple web UI powered by Streamlit
Optional detailed patch explanations (demo)
Requirements

Python 3.8+
PyTorch
MONAI
Streamlit
Install exact versions from requirements.txt.

Setup

# Clone this repository
git clone https://github.com/Galeophile/Benign-Malign-Tumor-Checker
cd Benign-Malign-Tumor-Checker

# Create and activate a virtual environment (macOS/Linux)
python -m venv monai_env
source monai_env/bin/activate

# Install dependencies
pip install -r requirements.txt
Run

Pick one of the available apps:

# Turkish UI
streamlit run tumor_detector_turkish.py

# Simple detector (confidence score)
streamlit run simple_tumor_detector.py

# Detector with patch explanations (demo)
streamlit run simple_explanation_detector.py

# Minimal test app
streamlit run simple_app.py
Then open http://localhost:8501 in your browser.

