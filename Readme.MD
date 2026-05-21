# Mobile Network Traffic Forecasting

## Assignment Overview

This project analyzes and forecasts mobile network traffic data from Milan, Italy, as part of a formative assignment for time series analysis and forecasting. The dataset covers 10,000 geographical squares (100×100 grid) over 61 days (November 1 – December 31, 2013) at 10-minute resolution.

**Key accomplishments:**
- Reduced 5GB raw data to 0.26GB (95% memory reduction)
- Conducted comprehensive EDA with 7 required analyses
- Implemented 3 forecasting models (1 statistical + 2 neural networks)

---

## Repository Structure
mobile-traffic-forecasting/
│
├── src/ # Source code
│ ├── data_handling.py # Task 1: Load & optimize 61 daily CSV files
│ ├── task2_eda.py # Task 2: All 7 EDA analyses
│ └── task3-MLP-shallow&DEEP.py # Task 3: ARIMA + MLP-Shallow + MLP-Deep
│
├── requirements.txt # Python dependencies
├── .gitignore # Excludes large data files
└── README.md # This file

text

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Telecom Italia Mobile (TIM) |
| Location | Milan, Italy |
| Grid size | 100 × 100 (10,000 squares) |
| Time period | November 1 – December 31, 2013 |
| Resolution | 10-minute intervals |
| Raw size | 5 GB (61 daily CSV files) |
| Processed size | 0.26 GB (Parquet) |

**Data access:**
- Raw data: [Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV)
- Processed Parquet: [Google Drive Link - Insert Your Link Here]

---

## Results Summary

### Task 1: Data Handling
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| File size | 5.0 GB | 0.26 GB | **95%** |
| Square id dtype | int64 (8 bytes) | category | 75% |
| Traffic dtype | float64 (8 bytes) | float32 (4 bytes) | 50% |

### Task 2: Exploratory Data Analysis

| Analysis | Key Finding |
|----------|-------------|
| PDF | Right-skewed distribution (skewness = 2.41) |
| Stationarity | ADF p-value = 0.0203 (stationary at 95% confidence) |
| Decomposition | Strong 24-hour seasonality |
| Anomalies | 18 detected, largest at Dec 16 15:00 (1166.49 units) |
| Spatial | Top hotspot: Square 6169 (1,626,388 units) |

### Task 3: Forecasting Models

**Models implemented:**
| Model | Type | Architecture |
|-------|------|--------------|
| ARIMA | Statistical | (5,1,3) |
| MLP-Shallow | Neural Network | Input(24) → Hidden(64) → Output(1) |
| MLP-Deep | Neural Network | Input(24) → Hidden(64,32,16) → Output(1) |

**Best performance (Square 4159):**
| Model | MAE | RMSE | MAPE | Train Time |
|-------|-----|------|------|-------------|
| ARIMA | 100,071 | 100,082 | 111,166% | 0.70s |
| **MLP-Shallow** | **81.84** | **91.58** | **104.3%** | **0.09s** |
| MLP-Deep | 84.51 | 97.14 | 109.5% | 0.08s |

**Failure analysis (Dec 16, 15:00 anomaly):**
| Square | Actual | MLP-Deep Prediction | Error |
|--------|--------|---------------------|-------|
| 6169 | 1166.49 | 487.29 | 58.2% |
| 4159 | 292.93 | 151.58 | 48.3% |
| 4556 | 226.97 | 223.79 | **1.4%** |

---

## Hardware & Software

| Component | Specification |
|-----------|---------------|
| OS | Windows 10 Pro |
| CPU | Intel Core i5-10210U @ 1.60GHz (4 cores, 8 threads) |
| RAM | 8.0 GB |
| Python | 3.12 |

**Key libraries:**
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- statsmodels (ARIMA, ADF test, decomposition)
- scikit-learn (MLP neural networks)

---

## Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/dahamkakooza/mobile-traffic-forecasting.git
cd mobile-traffic-forecasting
2. Install dependencies
bash
pip install -r requirements.txt
3. Download the data

## Data Setup

### Option A: Process raw data (takes longer)
1. Download raw CSV files from [Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV)
2. Place all 61 CSV files in `data/raw/`
3. Run `python src/data_handling.py`

4. Run the analysis
bash
# Task 1: Data handling (if using raw data)
python src/data_handling.py

# Task 2: Exploratory Data Analysis
python src/task2_eda.py

# Task 3: Forecasting models
python src/task3-MLP-shallow&DEEP.py
5. Output files
All outputs are saved to:

output/figures/ - 6+ plots (PDF, time series, heatmap, anomalies, etc.)

output/figures/task3/ - 3 prediction plots

output/tables/ - Performance tables and experiment logs

References
[1] G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri, "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, pp. 1-10, 2015. doi: 10.1038/sdata.2015.55

[2] Telecom Italia Mobile, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, 2015. doi: 10.7910/DVN/EGZHFV

[3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[4] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal approximators," Neural Networks, vol. 2, no. 5, pp. 359-366, 1989.

Author
Name: Kakooza Mahad
Course: Machine Learning Techniques I
