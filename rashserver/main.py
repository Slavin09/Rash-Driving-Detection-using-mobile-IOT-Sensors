# from flask import Flask, request, jsonify
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/uploadcsv", methods=["POST"])
# def upload_txt():
#     if "file" not in request.files:
#         return jsonify({"status": "error", "message": "No file part"}), 400

#     file = request.files["file"]
#     filename = file.filename or ""  # ensure it's a string

#     if filename == "":
#         return jsonify({"status": "error", "message": "No selected file"}), 400

#     base_filename, _ = os.path.splitext(filename)
#     save_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.csv")

#     file.save(save_path)

#     return jsonify({
#         "status": "ok",
#         "message": f"File received and saved as {save_path}"
#     })

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

#---------------------------------------

# from flask import Flask, request, jsonify
# import os

# app = Flask(__name__)

# UPLOAD_FILE = "imu_data.csv"

# @app.route('/uploadcsv', methods=['POST'])
# def upload_csv():
#     try:
#         data = request.get_data(as_text=True)
#         if not data.strip():
#             return jsonify({"status": "error", "message": "Empty data"}), 400

#         # Create or append to file
#         file_exists = os.path.exists(UPLOAD_FILE)
#         with open(UPLOAD_FILE, "a") as f:
#             if not file_exists:
#                 # write header only if first time (optional)
#                 f.write("device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n")
#             f.write(data.strip() + "\n")

#         return jsonify({"status": "ok", "message": "Data received"}), 200

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/')
# def index():
#     return "Server is running!"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)

# -------------------------------------

# from flask import Flask, request, jsonify
# import os
# import ast

# app = Flask(__name__)

# UPLOAD_FILE = "imu_data.csv"

# @app.route('/uploadcsv', methods=['POST'])
# def upload_csv():
#     try:
#         data = request.get_data(as_text=True)

#         if not data.strip():
#             return jsonify({"status": "error", "message": "Empty data"}), 400

#         # Clean up any JSON-like syntax if accidentally sent as a list
#         if data.startswith('[') and data.endswith(']'):
#             try:
#                 data_list = ast.literal_eval(data)
#                 data = "\n".join(data_list)
#             except Exception:
#                 pass

#         file_exists = os.path.exists(UPLOAD_FILE)
#         with open(UPLOAD_FILE, "a", encoding="utf-8") as f:
#             if not file_exists:
#                 f.write("device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n")
#             # Write clean data lines
#             f.write(data.strip() + "\n")

#         return jsonify({"status": "ok", "message": "Data received"}), 200

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/')
# def index():
#     return "Server is running!"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)

# --------------------------------------------------------

# from flask import Flask, request, jsonify
# import os
# import ast

# app = Flask(__name__)

# UPLOAD_FILE = "imu_data.csv"

# @app.route("/", methods=["GET"])
# def home():
#     return "Server is running!"

# @app.route("/uploadcsv", methods=["POST"])
# def upload_csv():
#     try:
#         # 1. Try to get from POST body raw
#         raw = request.get_data(as_text=True).strip()

#         # 2. If empty, try form data
#         if raw == "":
#             raw = request.form.get("text", "").strip()

#         # 3. If still empty, try request.values
#         if raw == "":
#             raw = request.values.get("text", "").strip()

#         # Debug print
#         # print("======== RAW RECEIVED =========")
#         # print(repr(raw))
#         # print("================================")

#         if raw == "":
#             return jsonify({"status":"error","message":"Empty data"}), 400

#         # Remove brackets and quotes if any
#         cleaned = raw.replace("[","").replace("]","").replace('"','')

#         # Split rows
#         rows = [r.strip() for r in cleaned.split("\n") if r.strip()]

#         # Write rows to CSV
#         file_exists = os.path.exists(UPLOAD_FILE)
#         with open(UPLOAD_FILE, "a") as f:
#             if not file_exists:
#                 f.write("device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n")
#             for r in rows:
#                 f.write(r + "\n")

#         return jsonify({"status":"ok","message":f"Received {len(rows)} rows"}), 200

#     except Exception as e:
#         print("SERVER ERROR:", e)
#         return jsonify({"status":"error","message":str(e)}), 500

# @app.route('/')
# def index():
#     return "Server is running!"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FINAL BLOCK
from flask import Flask, request, jsonify, send_file
import os
from predict import predict_rash_from_csv, model, scaler
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

UPLOAD_FILE = "imu_data.csv"
# ----------------------------------------


@app.route("/simulator")
def simulator():
    return send_file("index.html")


# ---------------------------------------


@app.route("/predict_window", methods=["POST"])
def predict_window():

    try:
        payload = request.get_json(
            force=True)  # robust and avoids "get is not a member" editors
        seq = payload.get("seq", []) if isinstance(payload, dict) else []

        # Validate
        if not seq or len(seq) < 50:
            return jsonify({"rash_probability": 0.0}), 200

        df = pd.DataFrame(seq)

        # Ensure required columns exist
        required = ["ax", "ay", "az", "azimuth", "pitch", "roll"]
        for c in required:
            if c not in df.columns:
                # if missing, return 0.0 (or you could raise)
                return jsonify({"rash_probability": 0.0}), 200

        df = df[required].astype(float)

        # scale
        scaled = scaler.transform(df.values)  # scaler from joblib
        X = np.expand_dims(scaled, axis=0)

        prob = float(model.predict(X)[0][1])  # probability of class 1 (rash)
        return jsonify({"rash_probability": prob}), 200

    except Exception as e:
        print("WINDOW ERROR:", e)
        return jsonify({"rash_probability": 0.0}), 200


# -----------------------------------------


@app.route("/predict", methods=["GET"])
def predict():
    try:
        probability = predict_rash_from_csv("imu_data.csv")
        return jsonify({"rash_probability": float(probability)}), 200
    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"rash_probability": 0.0}), 200


# -----------------------------------------
@app.route("/imu.csv", methods=["GET"])
def download_imu_csv():
    try:
        return send_file("imu_data.csv", mimetype="text/csv")
    except Exception as e:
        return f"CSV not found or unreadable: {e}", 404


def ensure_csv_header():
    header = "device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n"

    if not os.path.exists(UPLOAD_FILE) or os.path.getsize(UPLOAD_FILE) == 0:
        with open(UPLOAD_FILE, "w") as f:
            f.write(header)


@app.route("/", methods=["GET"])
def home():
    files = os.listdir(os.getcwd())
    html = "<h2>Server is running! <br> Files:</h2><ul>"
    for f in files:
        html += f'<li><a href="/{f}">{f}</a></li>'
    html += "</ul>"
    return html
    # return "Server is running!"


# -----------------------------------------------


# 2. Route to actually serve the file when clicked
@app.route('/<path:filename>')
def serve_file(filename):
    # send_from_directory is safer than send_file for variable paths
    return send_from_directory(os.getcwd(), filename)


# -----------------------------------------------


@app.route("/uploadcsv", methods=["POST"])
def upload_csv():
    try:
        raw = request.get_data(as_text=True).strip()

        if raw == "":
            raw = request.form.get("text", "").strip()

        if raw == "":
            raw = request.values.get("text", "").strip()

        if raw == "":
            return jsonify({"status": "error", "message": "Empty data"}), 400

        cleaned = raw.replace("[", "").replace("]", "").replace('"', "")

        rows = [r.strip() for r in cleaned.split("\n") if r.strip()]

        if len(rows) == 0:
            return jsonify({
                "status": "error",
                "message": "No valid rows"
            }), 400

        file_exists = os.path.exists(UPLOAD_FILE)

        with open(UPLOAD_FILE, "a") as f:
            if not file_exists:
                f.write(
                    "device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n"
                )

            for r in rows:
                r = r.strip()

                r = r.lstrip(",")

                r = r.rstrip(",")

                r = ",".join([x.strip() for x in r.split(",")])

                f.write(r + "\n")

        return jsonify({
            "status": "ok",
            "message": f"Received {len(rows)} rows"
        }), 200

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    ensure_csv_header()
    app.run(host='0.0.0.0', port=8080)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# from flask import Flask, request, jsonify, send_file
# import os
# import requests

# app = Flask(__name__)

# UPLOAD_FILE = "imu_data.csv"

# COLAB_PREDICT_URL = "https://4ff16d33310d.ngrok-free.app/predict"

# def ensure_csv_header():
#     header = "device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n"
#     if not os.path.exists(UPLOAD_FILE) or os.path.getsize(UPLOAD_FILE) == 0:
#         with open(UPLOAD_FILE, "w") as f:
#             f.write(header)

# @app.route("/")
# def home():
#     return "Server is running!"

# @app.route("/imu.csv", methods=["GET"])
# def download_imu_csv():
#     try:
#         return send_file("imu_data.csv", mimetype="text/csv")
#     except Exception as e:
#         return f"CSV not found or unreadable: {e}", 404

# @app.route("/uploadcsv", methods=["POST"])
# def upload_csv():
#     try:
#         raw = request.get_data(as_text=True).strip()

#         if raw == "":
#             raw = request.form.get("text", "").strip()

#         if raw == "":
#             raw = request.values.get("text", "").strip()

#         if raw == "":
#             return jsonify({"status": "error", "message": "Empty data"}), 400

#         cleaned = raw.replace("[", "").replace("]", "").replace('"', "")

#         rows = [r.strip() for r in cleaned.split("\n") if r.strip()]

#         if len(rows) == 0:
#             return jsonify({
#                 "status": "error",
#                 "message": "No valid rows"
#             }), 400

#         file_exists = os.path.exists(UPLOAD_FILE)

#         with open(UPLOAD_FILE, "a") as f:
#             if not file_exists:
#                 f.write("device_id,timestamp,ax,ay,az,azimuth,pitch,roll,lat,lon,speed\n")

#             for r in rows:
#                 r = r.strip()
#                 r = r.lstrip(",")
#                 r = r.rstrip(",")
#                 parts = [x.strip() for x in r.split(",")]

#                 f.write(",".join(parts) + "\n")

#                 try:
#                     data_row = {
#                         "ax": float(parts[2]),
#                         "ay": float(parts[3]),
#                         "az": float(parts[4]),
#                         "azimuth": float(parts[5]),
#                         "pitch": float(parts[6]),
#                         "roll": float(parts[7])
#                     }

#                     print("\n--- Sending to Colab ---")
#                     print("URL:", COLAB_PREDICT_URL)
#                     print("Payload:", data_row)

#                     res = requests.post(COLAB_PREDICT_URL, json=data_row, timeout=3)

#                     print("=== RESPONSE FROM COLAB ===")
#                     print("Status:", res.status_code)
#                     print("Body  :", res.text)
#                     print("=================================")

#                 except Exception as e:
#                     print("Prediction send error:", e)

#         return jsonify({
#             "status": "ok",
#             "message": f"Received {len(rows)} rows"
#         }), 200

#     except Exception as e:
#         print("SERVER ERROR:", e)
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     ensure_csv_header()
#     app.run(host='0.0.0.0', port=8080)
