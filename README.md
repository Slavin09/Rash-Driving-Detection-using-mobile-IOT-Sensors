# 🏍️ IMU-Based Kinematic Anomaly Detection for Two-Wheelers

> **Rash driving detection using smartphone IMU sensors, LSTM deep learning, and a real-time 3D visualization dashboard.**

---

## 📄 Paper

**Title:** Inertial Measurement Unit-Based Kinematic Anomaly Detection for Two-Wheelers  
**Conference:** IEEE Format  
**Institution:** Techno Main Salt Lake, Kolkata, India  
**Department:** CSE – Artificial Intelligence & Machine Learning  

| Role | Name | Email |
|------|------|-------|
| Author | Vinayak Puitandy | puitandyvinayak@gmail.com |
| Author | Saif Sahriar | saiffsahriar@gmail.com |
| Author | Sanket Manna | sanketm0406@gmail.com |
| Author | Vikiron Mondal | vikironmondal@gmail.com |
| Author | Saranya Adhikary | saranyaadhikary93@gmail.com |
| Mentor | Prof. Poojarini Mitra | p.mitra1.tmsl@ticollege.org |

---

## 📌 Overview

Road accidents involving two-wheelers account for over **30% of all fatal accidents in India** annually. This project presents a lightweight, IoT-driven system that uses the **IMU (Inertial Measurement Unit) sensors already present in any Android smartphone** to detect rash or anomalous riding behavior in real time — no dedicated hardware required.

The system captures triaxial accelerometer data alongside azimuth, pitch, and roll orientation, streams it to a Flask backend, and runs inference through a trained **LSTM / CNN-LSTM classifier**. A companion **Three.js browser simulator** visualizes the phone orientation in 3D and overlays the live rash probability score.

---

## 🏗️ System Architecture

```
┌─────────────────────┐        HTTP POST        ┌──────────────────────────┐
│   Android Client    │ ──────────────────────► │   Flask Server (Python)  │
│                     │   /uploadcsv            │                          │
│  • IMU @ ~10 Hz     │                         │  • Appends to CSV        │
│  • Accelerometer    │        HTTP POST        │  • /predict_window       │
│  • Gyroscope/Orient │ ◄────────────────────── │  • LSTM Inference        │
│  • GPS              │   rash_probability      │  • StandardScaler        │
└─────────────────────┘                         └──────────────────────────┘
                                                            │
                                                            │ serves
                                                            ▼
                                                ┌──────────────────────────┐
                                                │  Three.js Browser UI     │
                                                │                          │
                                                │  • 3D Cube Orientation   │
                                                │  • GPS Path Canvas       │
                                                │  • Rash % Overlay        │
                                                │  • CSV Playback/Scrub    │
                                                └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile Sensor Client | Android (IMU + GPS) |
| Backend Server | Python, Flask, Flask-CORS |
| ML Model | TensorFlow / Keras (LSTM, CNN-LSTM) |
| Preprocessing | NumPy, Pandas, Scikit-learn (StandardScaler) |
| Model Persistence | Joblib (scaler), HDF5 (model weights) |
| Frontend Visualization | Three.js, PapaParse, Vanilla JS |
| Data Format | CSV over HTTP |

---

## 📁 Project Structure

```
├── main.py                  # Flask server — data ingestion + inference endpoints
├── predict.py               # LSTM inference logic — loads model and runs prediction
├── index.html               # Three.js browser simulator and visualization dashboard
├── imu_data.csv             # Rolling sensor data file (auto-created on first upload)
├── model/
│   ├── rash_detection_lstm.h5    # Trained LSTM model weights
│   └── imu_scaler.pkl            # Fitted StandardScaler for feature normalization
└── README.md
```

---

## 🔌 API Endpoints

### `POST /uploadcsv`
Receives raw CSV rows from the Android client and appends them to `imu_data.csv`.

**Request body:** Plain text CSV rows (no header required after first write)
```
bike_v1,1700000000000,0.12,-0.05,9.81,45.2,3.1,-1.4,22.6148,88.4229,12.5
```

**Response:**
```json
{ "status": "ok", "message": "Received 1 rows" }
```

---

### `GET /predict`
Reads the last 50 rows from `imu_data.csv`, runs inference, and returns rash probability.

**Response:**
```json
{ "rash_probability": 0.73 }
```

---

### `POST /predict_window`
Accepts a 50-frame window directly as JSON for low-latency browser-side inference.

**Request body:**
```json
{
  "seq": [
    { "ax": 0.12, "ay": -0.05, "az": 9.81, "azimuth": 45.2, "pitch": 3.1, "roll": -1.4 },
    ...
  ]
}
```

**Response:**
```json
{ "rash_probability": 0.81 }
```

---

### `GET /simulator`
Serves the Three.js visualization frontend (`index.html`).

---

## 🧠 ML Model Details

### Input
- **Window size:** 50 consecutive timesteps
- **Features per timestep:** 6

| Feature | Description |
|---------|-------------|
| `ax` | Forward acceleration (m/s²) |
| `ay` | Lateral acceleration (m/s²) |
| `az` | Vertical acceleration (m/s²) |
| `azimuth` | Device yaw / heading (degrees) |
| `pitch` | Device pitch (degrees) |
| `roll` | Device roll (degrees) |

### Feature Vector
```
x_t = [ax, ay, az, azimuth, pitch, roll]
```

### Preprocessing
All six channels are standardized to **zero mean and unit variance** using a `StandardScaler` fitted on the training set and saved to `model/imu_scaler.pkl`.

### Output
```
p = P(rash | x_{t-49} ... x_t)  ∈ [0, 1]
```

---

## 📊 Model Comparison Results

| Model | Accuracy | AUC | Avg Precision |
|-------|----------|-----|---------------|
| XGBoost (DT Upgrade) | 82.9% | 0.917 | 0.906 |
| Tuned Random Forest | 84.6% | 0.922 | 0.903 |
| ResNet MLP | 84.4% | 0.909 | 0.898 |
| Bi-LSTM | 84.6% | 0.935 | 0.929 |
| **Ensemble CNN-LSTM** | **89.4%** | **0.955** | **0.952** |

The **Ensemble CNN-LSTM** outperforms all other architectures across every metric.

---

## 📈 Evaluation Plots

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)
> The CNN-LSTM records the fewest false negatives (26), making it the safest choice for a safety-critical application.

### ROC Curves
![ROC Curves](roc_curve.png)
> CNN-LSTM achieves AUC = 0.955, dominating at low false positive rates.

### Precision-Recall Curves
![Precision-Recall](precision_recall_curve.png)
> CNN-LSTM maintains near-perfect precision up to ~0.6 recall (AP = 0.952).

### Feature Importance
![Feature Importance](feature_importance.png)
> `ay_std` (lateral acceleration variability) and `ax_mean` (forward acceleration) are the two most discriminative features, confirming that **linear motion dynamics matter more than orientation alone**.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install flask flask-cors tensorflow keras joblib numpy pandas scikit-learn
```

### Run the Server
```bash
python main.py
```
Server starts at `http://0.0.0.0:8080`

### Open the Simulator
Navigate to:
```
http://localhost:8080/simulator
```

### Upload IMU Data
Post CSV rows from your Android app or any HTTP client:
```bash
curl -X POST http://localhost:8080/uploadcsv \
     -d "bike_v1,1700000000000,0.12,-0.05,9.81,45.2,3.1,-1.4,22.6148,88.4229,12.5"
```

---

## 🎮 Simulator Features

| Feature | Description |
|---------|-------------|
| **CSV Upload / URL Fetch** | Load recorded IMU sessions from file or remote URL |
| **Play / Pause / Scrub** | Frame-accurate playback with speed control (0.25× to 4×) |
| **3D Cube Orientation** | Real-time Euler angle rendering with selectable rotation order (ZYX / YXZ / XYZ) |
| **GPS Path Canvas** | 2D minimap of the ride trajectory with current position marker |
| **Rash Probability Overlay** | Live percentage updated every ~200 ms during playback |
| **Calibration** | Captures current orientation as baseline; persisted in `localStorage` |
| **Auto-Refresh** | Polls a CSV URL every 2 seconds for live field monitoring |

---

## ⚙️ CSV Data Format

```
device_id, timestamp, ax, ay, az, azimuth, pitch, roll, lat, lon, speed
```

| Column | Type | Description |
|--------|------|-------------|
| `device_id` | string | Device identifier |
| `timestamp` | long | Unix milliseconds |
| `ax` | float | Forward acceleration (m/s²) |
| `ay` | float | Lateral acceleration (m/s²) |
| `az` | float | Vertical acceleration (m/s²) |
| `azimuth` | float | Yaw / heading (degrees) |
| `pitch` | float | Pitch angle (degrees) |
| `roll` | float | Roll angle (degrees) |
| `lat` | float | GPS latitude |
| `lon` | float | GPS longitude |
| `speed` | float | Speed (m/s or km/h) |

---

## ⚠️ Known Limitations

- **Road vibration noise** — potholed surfaces can trigger false positives in the vertical acceleration channel
- **GPS dropout** — indoor or underground testing loses location data; the system degrades gracefully
- **Single-device dataset** — model weights are currently trained on one device and one rider; cross-device generalization is untested
- **HTTP upload latency** — on mobile networks, CSV rows may arrive delayed or out of order

---

## 🔭 Future Work

- **Federated Learning** — multi-rider model improvement without sharing raw sensor data
- **TensorFlow Lite** — on-device inference to eliminate server round-trip latency
- **Multi-class detection** — distinguish hard braking, sharp swerving, and sudden acceleration as separate event types
- **Fleet analytics dashboard** — aggregate rash episode frequency across multiple riders
- **Road safety heatmaps** — color GPS path segments by maximum observed rash probability

---

## 📜 License

This project is developed for academic research purposes under the guidance of **Prof. Poojarini Mitra**, Techno Main Salt Lake. Please cite the associated IEEE paper if you use this work.

---

> *"The barrier to deploying smartphone-based road safety systems at scale is lower than it appears — all you need is the phone already in the rider's pocket."*
