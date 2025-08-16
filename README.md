# WTI-Oil-Volatility-Forecast  
Replication of the methodology in *"Combining high frequency data with non-linear models for forecasting energy market volatility"* (Baruník & Krehlík, 2016, *Expert Systems with Applications*), applied to WTI crude oil futures data (2015–2024).  


## **Project Overview**  
This repository replicates the core framework from *barunik-0456185.pdf*, which combines high-frequency data with non-linear models (e.g., artificial neural networks) to forecast energy market volatility. We extend the analysis to a newer dataset: 1-minute, 5-minute, and daily prices of CL (WTI crude oil) futures traded on the New York Mercantile Exchange (NYMEX) from 2015 to 2024.  

Key goals:  
- Replicate the paper’s volatility forecasting pipeline (realized measures, models, and evaluation).  
- Validate whether findings (e.g., MedRV’s superiority, ANN’s statistical gains) hold for post-2015 WTI data.  


## **Dataset**  
- **Instrument**: CL (WTI Crude Oil) futures contracts (NYMEX).  
- **Time Range**: 2015–2024.  
- **Frequencies**: 1-minute, 5-minute, and daily data.  
- **Preprocessing**: Filtered to main trading hours; cleaned for missing values and outliers (see `src/data_processing/preprocess.py`).  

*Note: Raw data is not uploaded due to size. See `data/README.md` for sourcing details.*  


## **Methodology (Replicated from barunik-0456185.pdf)**  
### **Volatility Measures**  
Implemented key realized volatility estimators from the paper:  
- RV (Realized Variance)  
- RK (Realized Kernel)  
- TSRV (Two-Scale Realized Variance)  
- BV (Bipower Variation)  
- MedRV (Median Realized Volatility)  
- JWTSRV (Jump-Adjusted Wavelet Two-Scale Realized Variance)  

### **Forecasting Models**  
- Linear models: HAR (Heterogeneous Autoregressive), ARFIMA (Autoregressive Fractionally Integrated Moving Average).  
- Non-linear model: Feed-forward Artificial Neural Network (ANN).  
- Combination model: Linear blend of HAR and ANN forecasts.  

### **Evaluation**  
- Statistical metrics: RMSE, MAE, MME (to assess over/under-prediction).  
- Tests: MCS (Model Confidence Set) and SPA (Superior Predictive Ability) to compare model performance.  


## **Usage**  
### run py file in 02_code
