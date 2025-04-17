# Problem Statement:-
* Valuation and investment decisions in equity markets are often
influenced by fundamental indicators such as earnings, margins,
growth rates, and capital structure. While these indicators are
traditionally used by analysts to form qualitative judgments, the
opportunity to quantify their relationship with future market
performance using machine learning and deep learning opens new
frontiers in investment research.
In this challenge, participants will build deep learning models to
predict market capitalization growth of listed companies based on a
curated dataset of historical fundamental features. The objective is
to model complex, non-linear relationships between financial
indicators and future market valuation movements.

* Participants must use the provided dataset - consisting of time-
series fundamental financial indicators and market capitalization of
listed Indian companies - to build a predictive model for market cap
growth (targets).

## Time Series Forecasting with LSTM, GRU, and Transformer Models
This repository presents a comprehensive time series forecasting pipeline using advanced deep learning techniques, specifically LSTM, GRU, and Transformer-based models. The goal is to predict future values from structured time-dependent data using robust preprocessing and model training techniques.

## Dataset Overview
The dataset consists of multivariate time series data, processed to ensure:

* Missing value handling

* Feature normalization

* Time-based train-test splitting

* Windowing for supervised learning (using sliding windows approach)

* Data preprocessing is done in Data_Preprocessing.ipynb.

## Models Implemented
1. LSTM-GRU Hybrid Model
Notebook: LSTM_GRU_MODEL.ipynb

Combines the memory-retention capabilities of LSTM with the speed and efficiency of GRU layers.

## Model Architecture:

* Input Layer → LSTM Layer → GRU Layer → Dense Output Layer

* Incorporates EarlyStopping and ModelCheckpoint to prevent overfitting and retain the best-performing model.

* Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)

2. Transformer-based Forecasting Model
Notebook: FidelFolio_Transformer_Model.ipynb

Implements a Transformer Encoder architecture tailored for time series prediction.

Utilizes Positional Encoding and Multi-Head Attention to model long-term dependencies.

Training process includes:

Sliding windows for input sequences

Layer normalization

Custom training loops with loss visualization

#  Model Evaluation
Both models are evaluated on the test set using:

* Loss Curves for training and validation

* Performance Metrics: MSE, RMSE, MAE

* Forecast Plots: Actual vs. Predicted values

* The Transformer model demonstrated superior performance in capturing long-range patterns, while the LSTM-GRU hybrid provided strong performance with shorter sequence windows.
                       
# How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your_username/your_repo.git
cd your_repo
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run notebooks sequentially:

Data_Preprocessing.ipynb

LSTM_GRU_MODEL.ipynb or FidelFolio_Transformer_Model.ipynb

#  Future Improvements
Integrate exogenous features for improved context

Hyperparameter tuning using Optuna or Keras Tuner

Real-time inference dashboard

Experiment with N-BEATS or Temporal Fusion Transformers (TFT)

