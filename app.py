from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import requests
from datetime import datetime
import os

app = Flask(__name__)

# Load models & encoders
# Load trained models
rf_reg = joblib.load("models/best_rf_regressor.pkl")
rf_clf = joblib.load("models/best_xgb_classifier.pkl")

# Load encoders
le_city = joblib.load("models/city_encoder.pkl")
le_bucket = joblib.load("models/aqi_bucket_encoder.pkl")

# ✅ Features must match exactly what was used in train.py
feature_names = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene",
    "Hour", "Day", "Month", "Weekday", "City_enc"
]

# -----------------------
# Home Page
# -----------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------
# Option 1: CSV Upload
# -----------------------
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["file"]
        if not file:
            return render_template("result.html", prediction="❌ No file uploaded", title="CSV Prediction")

        df = pd.read_csv(file)

        # Ensure datetime parts
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Hour"] = df["Datetime"].dt.hour
        df["Day"] = df["Datetime"].dt.day
        df["Month"] = df["Datetime"].dt.month
        df["Weekday"] = df["Datetime"].dt.weekday

        # Safe city encoding
        known_cities = list(le_city.classes_)
        df["City_enc"] = df["City"].apply(
            lambda c: le_city.transform([c])[0] if c in known_cities else le_city.transform(["Other"])[0]
        )

        # Predictions
        X_input = df[feature_names]
        df["Predicted_AQI"] = rf_reg.predict(X_input)
        df["Predicted_Category"] = le_bucket.inverse_transform(rf_clf.predict(X_input))

        # Save file
        output_file = "predicted_results.csv"
        output_path = os.path.join("static", output_file)
        df.to_csv(output_path, index=False)

        # Convert to HTML table for preview
        table_html = df[["City", "Datetime", "Predicted_AQI", "Predicted_Category"]].head(20).to_html(classes="table table-striped", index=False)

        return render_template(
            "result.html",
            title="CSV Prediction",
            tables=[table_html],  # 👈 so {% if tables %} works
            download_link=output_file  # 👈 passes filename to template
        )

    except Exception as e:
        return render_template("result.html", prediction=f"❌ Error: {str(e)}", title="CSV Prediction")



# -----------------------
# Option 2: Manual Entry
# -----------------------
@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        # Collect user inputs
        data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Convert numeric fields
        numeric_features = [
            "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
            "SO2", "O3", "Benzene", "Toluene", "Xylene"
        ]
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Extract datetime parts
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Hour"] = df["Datetime"].dt.hour
        df["Day"] = df["Datetime"].dt.day
        df["Month"] = df["Datetime"].dt.month
        df["Weekday"] = df["Datetime"].dt.weekday

        # Safe city encoding
        known_cities = list(le_city.classes_)
        df["City_enc"] = df["City"].apply(
            lambda c: le_city.transform([c])[0] if c in known_cities else le_city.transform(["Other"])[0]
        )

        # Predictions
        X_input = df[feature_names]
        aqi_value = rf_reg.predict(X_input)[0]
        bucket_enc = rf_clf.predict(X_input)[0]
        bucket_label = le_bucket.inverse_transform([bucket_enc])[0]

        # Prepare preview table
        df["Predicted_AQI"] = aqi_value
        df["Predicted_Category"] = bucket_label
        table_html = df[["City", "Datetime", "Predicted_AQI", "Predicted_Category"]].to_html(classes="table table-striped", index=False)

        return render_template(
            "result.html",
            title="Manual Prediction",
            tables=[table_html],
            prediction=f"✅ Predicted AQI={aqi_value:.2f}, Category={bucket_label}"
        )

    except Exception as e:
        return render_template("result.html", prediction=f"❌ Error: {str(e)}", title="Manual Prediction")


# -----------------------
# Option 3: Real-Time API
# -----------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    city = request.form['city']
    token = "bebe8341a2a1cb07b9adf3d22e6f055fc82d82d0"  # replace with your token

    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url).json()

    if response.get("status") != "ok":
        return render_template(
            "result.html",
            prediction=f"❌ Could not fetch AQI for '{city}'. Reason: {response.get('data', 'Unknown error')}",
            title="Real-Time API Prediction"
        )

    aqi_value = response['data']['aqi']
    dominant = response['data']['dominentpol']
    timestamp = response['data']['time']['s']

    # Categorize AQI bucket (your own mapping)
    if aqi_value <= 50:
        bucket = "Good"
    elif aqi_value <= 100:
        bucket = "Moderate"
    elif aqi_value <= 200:
        bucket = "Unhealthy"
    elif aqi_value <= 300:
        bucket = "Very Unhealthy"
    else:
        bucket = "Hazardous"

    return render_template(
        "result.html",
        prediction=f"🌐 Live AQI for {city}: {aqi_value} ({bucket}), Dominant pollutant: {dominant}, Time: {timestamp}",
        title="Real-Time API Prediction"
    )


# -----------------------
# Download Predicted CSV
# -----------------------
@app.route("/download/<filename>")
def download_file(filename):
    filepath = os.path.join("static", filename)
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
