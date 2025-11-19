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
| Model              | RMSE ↓     | MAE ↓      | MAPE ↓ | sMAPE ↓ | Notes                                          |
| ------------------ | ---------- | ---------- | ------ | ------- | ---------------------------------------------- |
| **N-BEATS (Ours)** | ⭐ **Best** | ⭐ **Best** | Low    | Low     | Captures nonlinear patterns, structural breaks |
| SARIMA             | Medium     | Medium     | Higher | Higher  | Weak on multivariate data                      |
| Prophet            | Medium     | Higher     | Medium | Medium  | Good trends, weak noise handling               |

🧬 Dataset Characteristics

This project generates a rich, realistic synthetic dataset, including:

Polynomial + piecewise trend

Daily and weekly seasonalities

Structural breaks at random intervals

Outliers & heteroskedastic noise

Multiple exogenous variables

Example of generated target:

<p align="center"><img src="outputs/generated_series_sample.png" width="700"></p>

🔍 Decomposition Visualization
<p align="center"><img src="outputs/decomposition.png" width="750"></p>
Interpretation:

Blue → Model input history

Green/Red/Yellow layers → Individual block contributions

Sum → Final forecast

🧪 How N-BEATS Learns Internally
Trend Block

Learns long-term growth / drift.

Seasonal Block

Learns daily/weekly periodicities using sinusoidal basis functions.

Generic Block

Learns noise, outliers & complex nonlinear residual patterns.
                     ┌─────────────────────┐
                     │   Input Window (X)  │
                     │  [target + exogs]   │
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │  Flatten (MLP input)│
                     └─────────┬───────────┘
                               │
        ┌──────────────────────────────────────────────────────┐
        │                     N-BEATS STACKS                    │
        │                                                      │
        │   ┌─────────────┐   ┌──────────────┐   ┌──────────┐ │
        │   │ Trend Block │ → │ Seasonal Block│ → │ Generic │ │
        │   └──────┬──────┘   └─────┬────────┘   └────┬─────┘ │
        │          │                │                 │        │
        │     Backcast_i       Backcast_j       Backcast_k    │
        │         Forecast_i    Forecast_j   +  Forecast_k     │
        └────────────┬─────────────────────────────────────────┘
                      │
             ┌────────▼────────┐
             │  Final Forecast  │
             │    (24 steps)    │
             └──────────────────┘

