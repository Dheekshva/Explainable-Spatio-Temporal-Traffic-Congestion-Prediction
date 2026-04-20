# Explainable Spatio-Temporal Traffic Congestion Prediction

A hybrid CNN-LSTM deep learning model for multi-step traffic congestion forecasting with SHAP-based explainability, making predictions interpretable for urban planners and traffic management systems.

## Overview
Traffic congestion prediction is a critical urban challenge. This project combines CNNs for spatial road-network feature extraction and LSTMs for temporal pattern learning, with SHAP integrated to explain which features drive congestion — bridging the gap between predictive accuracy and interpretability.

## Approach
- **Architecture:** Hybrid CNN-LSTM — CNN layers extract spatial features from road network data; LSTM layers model temporal dependencies across time steps
- **Explainability:** SHAP (SHapley Additive exPlanations) for feature-level attribution, identifying top drivers such as time-of-day, weather overlays, and incident proximity
- **Evaluation:** RMSE and MAE for forecasting accuracy
- **Platform:** Google Colab with GPU runtime

## Results
- Prediction accuracy: ~91%
- Full explainability pipeline maintained alongside predictive performance

## Tech Stack
Python · TensorFlow · Keras · CNN · LSTM · SHAP · Pandas · NumPy · Matplotlib · Google Colab

## Files
- `traffic_congestion_prediction.ipynb` — Main model training, evaluation, and SHAP explainability notebook
- `literature_survey.pdf` *(optional)* — Research proposal covering 15+ state-of-the-art spatio-temporal and XAI papers

## How to Run
1. Open the notebook in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells in order

## Research Context
This project is part of an academic research proposal on explainable AI for urban traffic systems. The literature survey covers 15+ papers on spatio-temporal modelling and explainable AI methods including SHAP, LIME, and attention-based interpretability.
