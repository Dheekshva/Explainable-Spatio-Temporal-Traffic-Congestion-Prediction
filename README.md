# Diabetic Retinopathy Severity Grading — APTOS 2019

A deep learning model to classify the severity of diabetic retinopathy from retinal fundus images using EfficientNet-B4 with transfer learning and explainability via Grad-CAM.

## Overview
Diabetic retinopathy is a leading cause of blindness. Early and accurate grading is critical for timely treatment. This project builds an ordinal multi-class classifier on the APTOS 2019 Blindness Detection dataset, grading retinal images across 5 severity levels:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## Dataset
- **Source:** [APTOS 2019 Blindness Detection — Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **Size:** 3,662 training images
- **Format:** Fundus photographs (.png)

## Approach
- **Model:** EfficientNet-B4 pretrained on ImageNet, fine-tuned with differential learning rates
- **Class Imbalance:** Addressed using SMOTE oversampling
- **Evaluation Metric:** Cohen's Quadratic Weighted Kappa (consistent with Kaggle scoring)
- **Explainability:** Grad-CAM saliency maps to highlight pathological retinal regions
- **Platform:** Google Colab with GPU runtime

## Results
- Validation accuracy: ~88% (fine-tuning in progress with fully unfrozen backbone targeting 90%+)

## Tech Stack
Python · TensorFlow/Keras · EfficientNet-B4 · Grad-CAM · SMOTE · Pandas · NumPy · Matplotlib · Google Colab

## Files
- `diabetic_retinopathy_grading.ipynb` — Main training and evaluation notebook

## How to Run
1. Download the APTOS 2019 dataset from Kaggle
2. Upload to Google Drive
3. Open the notebook in Google Colab and mount Drive
4. Run all cells with GPU runtime enabled (Runtime → Change runtime type → T4 GPU)
