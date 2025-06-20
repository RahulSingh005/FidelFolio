# Market Capitalization Growth Forecasting with Deep Learning

This repository contains deep learning models (LSTM-GRU hybrid and Transformer) that predict future market capitalization growth of listed companies using historical fundamental indicators. The solution addresses the complex relationship between financial metrics and market valuation movements.

## Key Features
- **Multivariate Time Series Forecasting**: Processes historical fundamental indicators to predict market cap growth
- **Two Model Architectures**:
  - **LSTM-GRU Hybrid**: Balances long-term memory retention with computational efficiency
  - **Transformer Encoder**: Captures long-range dependencies using multi-head attention
- **Robust Preprocessing**: Automated handling of missing values, feature normalization, and time-based windowing

## Installation
1. Clone repository:
``
git clone https://github.com/your_username/your_repo.git
``
``
cd your_repo
``

3. Install dependencies:
   ``
   pip install -r requirements.txt
   ``

   
## Data Pipeline
- **Preprocessing**: `Data_Preprocessing.ipynb` handles:
  - Missing value imputation
  - Feature normalization
  - Time-based train-test splits
  - Sliding window sequence generation

## Model Training
| Model | Notebook | Key Components |
|-------|----------|----------------|
| LSTM-GRU Hybrid | `LSTM_GRU_MODEL.ipynb` | EarlyStopping, ModelCheckpoint, MSE/RMSE/MAE metrics |
| Transformer | `FidelFolio_Transformer_Model.ipynb` | Positional Encoding, Multi-Head Attention, Custom training loops |

## Results
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- **Performance Insights**:
  - Transformers excel at long-range pattern capture
  - LSTM-GRU performs better with shorter sequences

## How to Use
1. Preprocess data: Run `Data_Preprocessing.ipynb`
2. Train models: Execute either forecasting notebook
3. Evaluate: Check forecast plots (actual vs predicted) and metric logs

## Future Enhancements
- Incorporate exogenous economic indicators
- Hyperparameter tuning with Optuna
- Deployment dashboard for real-time inference
- Experiment with N-BEATS/Temporal Fusion Transformers
