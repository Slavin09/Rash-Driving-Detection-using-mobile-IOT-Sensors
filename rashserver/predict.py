# predict.py
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import os

TIME_STEPS = 50
FEATURE_COLS = ["ax","ay","az","azimuth","pitch","roll"]

# load
scaler = joblib.load("model/imu_scaler.pkl")
model = load_model("model/rash_detection_lstm.h5")

def predict_rash_from_csv(csv_path="imu_data.csv"):
    if not os.path.exists(csv_path):
        return 0.0

    df = pd.read_csv(csv_path)

    # require minimum 50 rows
    if len(df) < TIME_STEPS:
        return 0.0

    last = df.tail(TIME_STEPS)[FEATURE_COLS].values
    last_scaled = scaler.transform(last)

    X = np.expand_dims(last_scaled, axis=0)
    prob = model.predict(X)[0][1]  # probability of class "1" = rash

    return float(prob)
