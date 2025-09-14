# Air Quality Prediction 


## 📌 Project Overview
This project focuses on analyzing air quality data across multiple Indian cities.  
The dataset (`city_hour.csv`) contains hourly readings of pollutants and Air Quality Index (AQI) values.  
The final objective of the project is to clean, explore, and later build predictive models for AQI.

---
## Week 1


## ✅ Week 1 Goals
The Week 1 work mainly covered **Data Loading, Cleaning, and Exploratory Data Analysis (EDA)**.

### 1. Data Loading & Basic Checks
- Imported dataset using **pandas**.
- Performed initial checks:
  - `df.shape` → dataset dimensions
  - `df.info()` → datatypes & null values
  - `df.describe().T` → statistical summary for numerical columns
  - `df.isnull().sum()` → missing value counts

### 2. Advanced Cleaning & Handling Outliers
- Converted `Datetime` column to proper `datetime64[ns]` format.
- Dropped invalid datetime rows.
- Sorted data by **City** and **Datetime**.
- Capped unrealistic values:
  - Capped pollutant values (`PM2.5`, `PM10`, etc.) at valid thresholds.
  - Removed AQI values above 1000.
- Handled missing values:
  - **Time-based interpolation** (per city).
  - **Forward fill** + **Backward fill** for remaining NaNs.
- Removed duplicate rows.

### 3. Exploratory Data Analysis (EDA)
EDA was performed after cleaning to understand distributions, trends, and relationships:

- **AQI Distribution** → Visualized the overall AQI distribution across the dataset.
- **AQI Bucket Count** → Count of records in each AQI bucket (Good, Moderate, Poor, etc.).
- **City-wise Average AQI (Top 15 cities)** → Compared average AQI levels across cities.
- **Trend over Time (Delhi)** → Visualized AQI monthly trend for Delhi (2015–2020).
- **Correlation Heatmap** → Displayed correlations between pollutants and AQI.
- **Boxplots (after cleaning)** → Checked pollutant distributions to assess outliers and spread.

---

## 📊 Key Insights (Week 1)
- Dataset had significant missing values, successfully interpolated.
- Outliers (extreme sensor values) were capped for realistic analysis.
- AQI varies significantly across cities, with the top 15 showing high pollution load.
- Clear monthly AQI variation was observed for Delhi between 2015–2020.
- Pollutants like **PM2.5** and **PM10** show strong correlation with AQI.

---
## ⚠️ Dataset Information  
The dataset used in this project is large and cannot be uploaded directly to GitHub.  
👉 **Please download the dataset from Kaggle**:  
[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  

Once downloaded, place the CSV file in the project directory before running the notebook.  

---

## Week 2

## ✅ Week 2 Goals
- Perform **feature engineering** to prepare dataset for machine learning.
- Build **regression models** for predicting AQI numeric value.
- Build **classification models** for predicting AQI category (Good, Moderate, Poor, etc.).
- Apply **hyperparameter tuning** (optimized with smaller grids & fewer folds).
- Add **visual evaluation tools** for model interpretation.
- Save best models for deployment in Flask (Week 3).

---

## 🔹 Work Completed

### 1. Feature Engineering
- Extracted **time-based features**: Hour, Day, Month, Weekday.
- Created **lag features** (previous AQI, PM2.5).
- Generated **rolling averages (3h)** for pollutants.
- Encoded **City column** using Label Encoding.
- Dropped NaN rows introduced by lag/rolling operations.

### 2. Dataset Sampling
- Original dataset was very large (~1M+ rows).
- Used **20% random sample (~200k rows)** for faster training and tuning.
- Preserved data diversity across multiple cities and time periods.

---

## 🔹 Model Building

### Regression Models (Predict AQI Value)
| Model               | RMSE (↓ Better) |
|----------------------|-----------------|
| Linear Regression    | **14.90**       |
| Random Forest        | **13.58**       |
| XGBoost              | **13.93**       |

✅ **Best model** → Random Forest Regressor (lowest error).

📌 **Interpretation**: Predictions are typically within ±14 AQI units of the true value.

---

### Classification Models (Predict AQI Category)
| Model               | Accuracy (↑ Better) |
|----------------------|---------------------|
| Logistic Regression  | **66.8%**           |
| Random Forest        | **94.8%**           |
| XGBoost              | **94.3%**           |

✅ **Best model** → Random Forest Classifier (slightly better than XGBoost).

📌 **Interpretation**: AQI categories were predicted correctly in ~95% of test cases.

---

## 🔹 Hyperparameter Tuning
- Used **RandomizedSearchCV** instead of GridSearch for speed.
- Tuned only on **sampled dataset (20%)**.
- Limited parameter grids to 2–3 values each.
- Used **fewer folds (cv=2)** for quicker evaluation.

**Optimized Models:**
- Random Forest Regressor (Regression) – Best Params selected.
- XGBoost Classifier (Classification) – Best Params selected.

---

## 🔹 Visual Evaluation Tools

### 1. Residual Plot – Regression
- Showed error distribution around predicted AQI.
- Confirmed that Random Forest predictions were centered with fewer extreme errors.

### 2. Confusion Matrix – Classification
- Demonstrated **how accurately each AQI category was predicted**.
- Showed Random Forest & XGBoost both achieved ~95% accuracy.

### 3. Feature Importance
- Random Forest & XGBoost highlighted **PM2.5, PM10, NO2, SO2, CO, O3** as the most critical features.
- Time features (Hour, Month, Weekday) also contributed.

---

## 🔹 Model Saving
- **Saved models and encoders for Flask deployment**:
  - `best_rf_regressor.pkl`
  - `best_xgb_classifier.pkl`
  - `city_encoder.pkl`
  - `aqi_bucket_encoder.pkl`

---

## 📊 Results & Insights
- **Regression**: AQI prediction within ±14 units → good precision for pollution forecasting.
- **Classification**: AQI category prediction ~95% accuracy → highly reliable for issuing alerts.
- Tree-based models (Random Forest, XGBoost) **outperformed linear models** by a wide margin.


---

