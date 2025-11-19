# Ai-project-cultus
This project implements a production-grade N-BEATS deep learning model for advanced time-series forecasting, alongside traditional benchmark models and interpretability tools. It is designed for students, researchers, and practitioners working on real-world temporal prediction tasks involving trend, seasonality, noise, and structural shifts 
📌 Overview

This repository contains a complete, production-quality implementation of the N-BEATS deep learning architecture for advanced time-series forecasting.

Built from scratch in PyTorch, this project includes:

✔ Complex multivariate dataset generation
✔ Multi-step N-BEATS forecasting
✔ Interpretability via basis decomposition
✔ SARIMA & Prophet benchmarking
✔ GPU-ready Google Colab notebook
✔ Ablation testing
✔ Rolling-origin cross-validation

This project is ideal for:

ML & AI Researchers

Data Scientists

University Coursework / Capstone Projects

Forecasting Engineers

Anyone learning deep time-series modeling

🚀 Key Features
🔥 Deep Learning

Full N-BEATS architecture

Multivariate input support

Multi-step forecasting

Trend + seasonal basis functions

Gradient clipping, AdamW, CosineLR

📊 Statistical Benchmarks

SARIMA (Statsmodels)

Prophet (Meta)

Performance comparison using:
 * RMSE

 * MAE

* MAPE

* sMAPE

* MASE
 

🧠 Explainability

Block-wise Backcast & Forecast decomposition

Trend vs Seasonality separation

Residual block interpretation

Ablation analysis (removing blocks & measuring impact)

🧪 Evaluation Framework

Weighted rolling-origin evaluation

Train/validation/test split based on walk-forward logic

Synthetic + reproducible time-series patterns
