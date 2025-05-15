from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("humidity_rf_model.pkl")

# Load encoder nếu có
# encoder = joblib.load("encoder.pkl")  # Nếu bạn có file encoder riêng

def get_part_of_day(hour):
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        timestamp = pd.to_datetime(data["timestamp"])

        hour = timestamp.hour
        dayofweek = timestamp.dayofweek
        month = timestamp.month
        part_of_day = get_part_of_day(hour)

        # One-hot encoding thủ công
        part_cols = ["part_of_day_afternoon", "part_of_day_evening", "part_of_day_morning", "part_of_day_night"]
        encoded = {col: 0 for col in part_cols}
        encoded[f"part_of_day_{part_of_day}"] = 1

        input_data = {
            "hour": hour,
            "dayofweek": dayofweek,
            "month": month,
            **encoded
        }

        df = pd.DataFrame([input_data])
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(df)[0]
        return jsonify({"predicted_humidity": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Humidity Prediction API is running."
