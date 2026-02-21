# VoltVision – Multi-Variate Energy Consumption Forecasting System

## Overview
VoltVision is a containerized machine learning project designed to forecast multi-variate energy consumption using both deep learning and traditional time-series methods.  
The goal of this project is to simulate a real-world forecasting pipeline that includes data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and automated deployment using Docker.

The system compares a deep learning model (LSTM) with a traditional baseline model and generates forecast outputs along with evaluation metrics and visualizations.

This project demonstrates a full end-to-end machine learning workflow suitable for applications in energy, finance, and retail forecasting scenarios.

---

## Key Features
- Multi-variate time series preprocessing  
- Feature engineering with lag and rolling statistics  
- Deep learning forecasting using LSTM (PyTorch)  
- Traditional baseline model using Prophet  
- Hyperparameter tuning with Optuna  
- Experiment tracking with Weights & Biases (offline mode)  
- Walk-forward validation strategy  
- Fully containerized pipeline using Docker  
- Automatic generation of metrics, forecasts, and plots  

---

## Project Structure
VoltVision/

├── data/

│ ├── raw/

│ └── processed/

├── logs/

├── results/

├── src/

│ ├── preprocess.py

│ ├── feature_engineering.py

│ ├── models.py

│ ├── train.py

│ └── evaluate.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── .env.example

└── README.md



---

## Dataset
This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

Dataset source:  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains minute-level measurements of household electric power consumption and multiple related variables.  
During preprocessing, the data is cleaned, resampled to hourly intervals, and normalized.

---

## Methodology

### 1. Data Preprocessing
- Handle missing values  
- Convert datetime fields  
- Resample to hourly frequency  
- Normalize features  
- Save processed dataset for modeling  

### 2. Feature Engineering
- Calendar features (hour, day of week, month)  
- Lag features (previous values)  
- Rolling mean features  
- Final dataset saved for training  

### 3. Models
**Deep Learning Model**
- LSTM neural network implemented using PyTorch  
- Captures sequential dependencies in time-series data  

**Baseline Model**
- Prophet forecasting model used for comparison  

### 4. Training Pipeline
- Walk-forward validation approach  
- Hyperparameter tuning using Optuna  
- Experiment tracking using Weights & Biases (offline mode)  
- Training logs saved to `logs/training.log`  

### 5. Evaluation
The evaluation script generates:
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- MAPE (Mean Absolute Percentage Error)  
- Prediction intervals  

Outputs are saved in the `results/` directory.

---

## Output Files

After running the pipeline, the following files are generated:

results/

├── metrics.json

├── forecasts.csv

└── forecast_visualization.png


logs/

└── training.log


**metrics.json**  
Contains performance metrics for both deep learning and baseline models.

**forecasts.csv**  
Contains timestamps, actual values, predictions, and prediction intervals.

**forecast_visualization.png**  
Plot showing actual vs predicted values with confidence bounds.

---

## How to Run the Project

### Requirements
- Docker Desktop installed  
- Git installed  

### Steps

Clone the repository:
git clone https://github.com/shahid200620/VoltVision.git
cd VoltVision

Run the full pipeline:


docker-compose up --build


This single command will:
- Preprocess the dataset  
- Generate features  
- Train models  
- Perform hyperparameter tuning  
- Evaluate results  
- Save outputs  

No manual steps required after running the command.

---

## Environment Variables

An example environment file is provided:
.env.example


Variables included:
WANDB_API_KEY=your_wandb_api_key_here
DATASET_URL=https://archive.ics.uci.edu/static/public/235/household_power_consumption.zip


For this project, Weights & Biases runs in offline mode by default.

---

## Walk-Forward Validation

The training process uses walk-forward validation, which simulates real-world forecasting.  
The model is repeatedly trained on earlier data and evaluated on future segments.  
This approach ensures that time-order is respected and avoids data leakage.

Training progress is logged in:
logs/training.log


---

## Technologies Used
- Python  
- PyTorch  
- Prophet  
- Pandas  
- Scikit-learn  
- Optuna  
- Weights & Biases  
- Matplotlib  
- Docker  

---

## Notes
This project focuses on building a reproducible forecasting pipeline rather than optimizing for maximum accuracy.  
The goal is to demonstrate system design, experiment tracking, and containerized deployment for time-series forecasting.

---

## Author
Shahid Mohammed  

Global Placement Program – Data Science Track

---

## Running Again
To rerun the entire pipeline:

docker-compose up --build

All results will regenerate automatically.
