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
