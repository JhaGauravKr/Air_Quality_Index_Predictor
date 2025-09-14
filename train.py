import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("city_hour.csv")

# 2. Preprocess datetime
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["Weekday"] = df["Datetime"].dt.weekday

# 3. Encode city
# le_city = LabelEncoder()
# df["City_enc"] = le_city.fit_transform(df["City"])
all_cities = df["City"].unique().tolist() + ["Other"]
le_city = LabelEncoder()
le_city.fit(all_cities)
df["City_enc"] = le_city.transform(df["City"])

# 4. Encode AQI_Bucket
le_bucket = LabelEncoder()
df["AQI_Bucket_enc"] = le_bucket.fit_transform(df["AQI_Bucket"].astype(str))

# 5. Define features
features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene",
    "Hour", "Day", "Month", "Weekday", "City_enc"
]

# 6. Handle missing values
df = df.dropna(subset=["AQI"])
df[features] = df[features].fillna(df[features].median())

# 7. 🔥 Sample smaller dataset for faster training
df = df.sample(n=100000, random_state=42)   # 100k rows

# 8. Split dataset
X = df[features]
y_reg = df["AQI"]
y_clf = df["AQI_Bucket_enc"]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# 9. Train models (faster settings)
print("Training RandomForest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)

print("Training RandomForest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_clf_train)

print("Training XGBoost Classifier...")
xgb_clf = XGBClassifier(
    n_estimators=50, learning_rate=0.1, max_depth=6, subsample=0.5,
    random_state=42, n_jobs=-1
)
xgb_clf.fit(X_train, y_clf_train)

# 10. Save models & encoders
joblib.dump(rf_reg, "models/best_rf_regressor.pkl")
joblib.dump(rf_clf, "models/best_rf_classifier.pkl")
joblib.dump(xgb_clf, "models/best_xgb_classifier.pkl")
joblib.dump(le_city, "models/city_encoder.pkl")
joblib.dump(le_bucket, "models/aqi_bucket_encoder.pkl")
joblib.dump(features, "models/feature_names.pkl")

print("✅ Training complete. Models saved in /models folder.")
