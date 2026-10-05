<h1 align="center">🏍️ IMU-based Kinematic Anomaly Detection for Two-Wheelers</h1>

<p align="center">
  <i>Catching rash riding in real time using nothing but the sensors inside a regular Android phone.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-89.4%25-brightgreen?style=for-the-badge" alt="Accuracy" />
  <img src="https://img.shields.io/badge/AUC-0.955-blue?style=for-the-badge" alt="AUC" />
  <img src="https://img.shields.io/badge/Avg%20Precision-0.952-orange?style=for-the-badge" alt="AP" />
  <img src="https://img.shields.io/badge/Latency-35%E2%80%9380%20ms-purple?style=for-the-badge" alt="Latency" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Flask-000000?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Keras-LSTM-D00000?logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Three.js-000000?logo=threedotjs&logoColor=white" alt="Three.js" />
  <img src="https://img.shields.io/badge/Android-3DDC84?logo=android&logoColor=white" alt="Android" />
  <img src="https://img.shields.io/badge/Paper-IEEE%20Conference-00629B" alt="IEEE" />
</p>

<p align="center">
  <b>Vinayak Puitandy</b> · <b>Saif Sahriar</b> · <b>Sanket Manna</b> · <b>Vikiron Mondal</b> · <b>Saranya Adhikary</b> · <b>Poojarini Mitra</b>
  <br/>
  Dept. of CSE (AI &amp; ML) / CSE (Data Science), Techno Main Salt Lake, Kolkata, India
</p>

---

## 📑 Table of Contents

- [📝 Abstract](#-abstract)
- [🎯 Highlights](#-highlights)
- [🧭 Introduction](#-introduction)
- [📚 Related Work](#-related-work)
- [🏗️ System Architecture](#️-system-architecture)
- [🧪 Data Pipeline and Preprocessing](#-data-pipeline-and-preprocessing)
- [📊 Model Evaluation](#-model-evaluation)
- [🔬 Experimental Setup](#-experimental-setup)
- [⚠️ Challenges and Limitations](#️-challenges-and-limitations)
- [🚀 Future Work](#-future-work)
- [✅ Conclusion](#-conclusion)
- [📖 Citation](#-citation)
- [🔗 References](#-references)

---

## 📝 Abstract

> Two-wheeler accidents kill thousands of people on Indian roads every year, and a large chunk of those crashes trace back to reckless riding. We built a system that tries to catch this behavior in real time using nothing more than the **Inertial Measurement Unit (IMU)** sensors sitting inside a regular Android phone. The phone records acceleration and orientation data as the rider moves, ships those readings to a small **Flask** server over HTTP, and the server runs a trained **LSTM** model on a rolling window of the last fifty frames to decide whether the current riding pattern looks normal or rash.
>
> On top of that, we put together a browser-based **3D simulator using Three.js** that lets us watch the phone orientation replay live and see the rash probability update as playback moves forward. We trained and compared five different classifiers on the same dataset, and the **CNN-LSTM ensemble came out on top with 89.4% accuracy and an AUC of 0.955**. The broader point we are trying to make is that you do not need specialized hardware to do this kind of monitoring. The phone in the rider's pocket is enough.

**Index Terms:** `Inertial Measurement Unit` · `rash driving detection` · `CNN-LSTM` · `IoT` · `two-wheeler safety` · `anomaly detection`

---

## 🎯 Highlights

| | |
|---|---|
| 📱 **No extra hardware** | Runs on the six-axis IMU already inside a standard Android phone |
| ⚡ **Real time** | 35 to 80 ms inference per window, well under the 200 ms polling interval |
| 🧠 **Five classifiers compared** | XGBoost, Random Forest, ResNet MLP, Bi-LSTM, Ensemble CNN-LSTM |
| 🏆 **Best model** | Ensemble CNN-LSTM: **89.4%** accuracy, **0.955** AUC, **0.952** AP |
| 🛡️ **Safety first** | Only **26** missed rash episodes (false negatives), versus 42 for Bi-LSTM and 52 for XGBoost |
| 🎮 **3D debugging tool** | Three.js simulator replays phone orientation with a live rash probability overlay |

---

## 🧭 Introduction

If you look at road accident statistics for India, two-wheelers show up in a disproportionately large share of the numbers. The Ministry of Road Transport and Highways puts the figure at over **30% of all fatal accidents annually**, and rash driving consistently appears near the top of the listed causes [[1]](#-references). This is not a new observation, but it has proven remarkably hard to act on. Telling people to ride carefully does not scale. What might scale, at least in theory, is automated monitoring.

The conventional hardware-based approaches to driving behavior monitoring (OBD dongles, dedicated accelerometer units, dashboard cameras with vision-based pipelines) tend to either cost too much or require installation effort that most two-wheeler riders are simply not going to bother with [[2]](#-references). Then it occurred to us that almost every rider in urban India already carries a six-axis IMU in their shirt pocket. A modern smartphone has an accelerometer, a gyroscope, and orientation sensors that together give a surprisingly complete picture of how the device, and by extension the rider, is moving through space.

This project grew out of that observation. We wanted to see whether a smartphone-based pipeline, from raw sensor collection through to a trained deep learning classifier, could actually produce reliable rash detection without any additional hardware. The system has three parts:

1. 📱 An **Android app** that samples IMU data at roughly 10 Hz and streams it to a server.
2. 🖥️ A **Python Flask backend** that accumulates those readings and runs inference using an LSTM model.
3. 🎮 A **Three.js visualization tool** that proved invaluable during testing for understanding what the model was actually responding to.

### Contributions

- ✅ A working **end-to-end rash detection pipeline** that runs entirely on commodity hardware, with no sensors beyond what a standard Android phone already has.
- ✅ A **sliding-window LSTM and CNN-LSTM architecture** designed for real-time classification of six-channel IMU streams.
- ✅ A **rigorous comparison of five classifiers** (XGBoost, Random Forest, ResNet MLP, Bi-LSTM, and an Ensemble CNN-LSTM) evaluated on the same data using confusion matrices, ROC curves, and precision-recall analysis.
- ✅ A **Three.js 3D visualization tool** that turned out to be genuinely useful for debugging model behavior in the field.

---

## 📚 Related Work

Using onboard sensors to infer how someone is driving is not a new idea. Johnson and Trivedi [[3]](#-references) showed over a decade ago that you can pick out events like hard braking or aggressive cornering from accelerometer data alone, using nothing more sophisticated than well-chosen thresholds. That work established a baseline, but threshold methods have a well-known fragility problem: the cutoffs that work on one vehicle type or road condition often fail on another.

This pushed researchers toward models that could learn the relevant patterns from data. SVMs applied to windowed accelerometer statistics were a popular middle ground for a while, and Eren et al. [[4]](#-references) explored this approach with reasonable results on passenger car datasets. The limitation was always feature engineering: you had to decide ahead of time which statistical summaries of the window mattered, and that decision carried many implicit assumptions about what rash behavior looks like in sensor space.

Recurrent architectures, and LSTMs in particular, changed the picture because they can learn temporal structure directly from raw sequences rather than hand-crafted summaries [[5]](#-references). For driving behavior, where what happened in the last two or three seconds of motion often matters as much as the current reading, this is a meaningful advantage.

Surprisingly little of this work has focused specifically on two-wheelers. Saiprasert et al. [[6]](#-references) studied smartphone sensor data from motorcycles, though their focus was road surface quality rather than rider behavior. The motorcycle context is genuinely different from cars: riders lean into turns, weight shifts are more dramatic, and the vibration profile of a two-stroke engine looks nothing like a passenger sedan. We tried to keep these differences in mind during data collection and modeling, even though the current dataset is still limited in scope.

---

## 🏗️ System Architecture

The system is split into three layers that talk to each other over standard HTTP. Nothing exotic about any individual component; the interesting part is how they fit together to produce a low-latency inference loop.

```mermaid
flowchart LR
    A["📱 Android App<br/>IMU @ ~10 Hz"] -->|"CSV batches<br/>over HTTP"| B["🖥️ Flask Server<br/>CORS enabled"]
    B --> C[("📄 Stored CSV")]
    C -->|"GET /predict<br/>last 50 rows"| D["🧠 Keras Model<br/>LSTM / CNN-LSTM"]
    E["🎮 Three.js Simulator<br/>Browser"] -->|"POST /predict_window<br/>50 frames as JSON"| D
    D -->|"rash_probability"| E
    D -->|"rash_probability"| B
```

### 📱 A. Mobile Client and Sensor Stream

The Android side reads from the phone's built-in IMU at approximately **10 Hz**. At each sample it records:

- Three axes of linear acceleration (`ax`, `ay`, `az`) in m/s²
- Device orientation as azimuth, pitch, and roll in degrees
- GPS latitude and longitude when available
- A Unix millisecond timestamp

These readings are formatted as **CSV rows** and batched into small payloads before being posted to the server. The upload format is deliberately simple (plain text, no JSON wrapping at the row level) to minimize per-row serialization overhead on the phone and to make raw uploads easy to inspect during debugging.

### 🖥️ B. Flask Server and Inference Endpoints

The backend runs on Flask with CORS enabled, which became necessary once we started hitting the server from a browser-based client. It exposes two inference paths that ended up serving slightly different purposes.

| Endpoint | Method | Input | Purpose |
|---|---|---|---|
| `/predict` | `GET` | Reads the last 50 rows of the stored CSV | Monitoring live sensor data actively uploaded from a phone |
| `/predict_window` | `POST` | JSON array of 50 frames sent by the client | Main path during playback and replay; skips CSV writes entirely |

`/predict` opens the stored CSV, reads the last fifty rows, scales the six feature columns using a pre-fitted `StandardScaler` loaded at startup, and passes the resulting `(1 × 50 × 6)` tensor to the Keras model. The response is a JSON object with a `rash_probability` field.

`/predict_window` was added once we realized the browser simulator needed to trigger predictions from its in-memory parsed data without waiting for CSV writes to complete. In practice, this became the main inference path during playback and replay testing.

### 🧠 C. LSTM Model Architecture

Each sample fed to the classifier is a tensor of shape `(50, 6)`: fifty timesteps of a six-dimensional feature vector. Formally, the feature vector at time $t$ is:

$$
\mathbf{x}_t = [a_x,\ a_y,\ a_z,\ \theta_{az},\ \theta_{pitch},\ \theta_{roll}]
$$

All channels are standardized to zero mean and unit variance. The scaler is fitted on the training set and frozen before deployment, so the same transformation is applied at inference time. The model produces a single output probability:

$$
p = P\left(\text{rash} \mid \mathbf{x}_{t-49:t}\right)
$$

Training used **binary cross-entropy** loss with **Adam**. Model weights are saved in HDF5 format and the scaler as a joblib pickle, both loaded once at server startup to avoid repeated disk reads during inference.

### 🎮 D. Three.js Visualization Frontend

The browser-side visualization began as a debugging tool and because we wanted to see what the sensor data looked like in three dimensions.

- 📦 Renders a box roughly shaped like a phone and updates its Euler angles in real time as playback advances, using calibration-adjusted pitch, yaw, and roll.
- 🗺️ A small canvas below the 3D view draws the **GPS trajectory** of the session and highlights the current position.
- 🔄 The rotation order is **user-selectable (ZYX, YXZ, or XYZ)**, since different phone mounting orientations required different Euler conventions to look correct.
- 🎯 A calibration button captures the current orientation as a baseline offset, stored in `localStorage` so it survives page refreshes.
- 📈 The rash probability percentage updates **every 200 ms** in the corner of the viewport, frequent enough to follow the model's response to specific maneuvers without hammering the server.

---

## 🧪 Data Pipeline and Preprocessing

### A. CSV Format and Header Handling

Every row stored on the server follows this schema:

```csv
device_id, timestamp, ax, ay, az, azimuth, pitch, roll, lat, lon, speed
```

The browser parser does flexible header matching. It looks for column names case-insensitively and accepts common synonyms like `yaw` in place of `azimuth`. If no header row is found at all, it falls back to positional reading, treating columns five through seven as the orientation channels. This was necessary because early Android builds produced slightly inconsistent headers across different devices.

### B. Sliding Window Construction

The inference window ending at frame index $i$ is:

$$
W_i = \{\mathbf{x}_{\max(0,\,i-49)},\ \dots,\ \mathbf{x}_i\}
$$

- When $i < 49$, the window pads from the start of the recording rather than failing.
- Missing channel values default to zero.
- During playback the window advances by one frame every five animation frames, which works out to roughly **one inference call every 200 ms** at normal speed.

### C. Orientation Calibration

A phone's raw orientation readings depend entirely on how it is mounted on the bike. A phone sitting flat in a tank bag produces completely different azimuth and pitch values than the same phone clipped vertically to the handlebars, even if the bike is doing identical things in both cases. We handle this by subtracting a baseline offset $(\delta_{yaw}, \delta_{pitch}, \delta_{roll})$ captured while the rider is stationary in their intended riding position:

$$
\theta'_{*} = \theta_{*} - \delta_{*}
$$

The corrected angles $\theta'$ feed both the 3D renderer and the inference pipeline. Calibration values persist across sessions via `localStorage`.

---

## 📊 Model Evaluation

### A. Classifier Comparison

Five models were trained on the same dataset and evaluated on a held-out validation split.

| Model | Accuracy | AUC | AP |
|:---|:---:|:---:|:---:|
| XGBoost (DT Upgrade) | 82.9% | 0.917 | 0.906 |
| Tuned Random Forest | 84.6% | 0.922 | 0.903 |
| ResNet MLP | 84.4% | 0.909 | 0.898 |
| Bi-LSTM | 84.6% | 0.935 | 0.929 |
| 🏆 **Ensemble CNN-LSTM** | **89.4%** | **0.955** | **0.952** |

The CNN-LSTM ensemble beat every other architecture on all three metrics by a noticeable margin. The gap between it and the Bi-LSTM is roughly five percentage points in accuracy and two points in AUC, which is meaningful for a safety application. The tree-based models were not far behind in accuracy but lagged considerably on AUC and AP, suggesting they struggle more with the harder boundary cases between normal and rash behavior.

### B. Confusion Matrices

<p align="center">
  <img width="1600" height="320" alt="image" src="https://github.com/user-attachments/assets/a264a0d4-20b8-42ed-90bf-35561b39f8bc" />
</p>
<p align="center"><sub><b>Fig. 1.</b> Confusion matrices across all five classifiers. The CNN-LSTM (rightmost) records the highest true positive count at 183 and the fewest missed rash episodes at 26.</sub></p>

The raw confusion numbers show why the CNN-LSTM matters for this application. It produced only **26 false negatives** (cases where a rash episode was missed entirely), compared to **52 for XGBoost** and **42 for the Bi-LSTM**. In a safety monitoring context, a missed detection is a worse outcome than a false alarm, so this is arguably more important than the raw accuracy number.

### C. ROC Curves

<p align="center">
  <img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/48e71a27-23ff-477b-976e-b1c7e38414e4" />
</p>
<p align="center"><sub><b>Fig. 2.</b> ROC curves for the five optimized models. CNN-LSTM leads with AUC = 0.955; Bi-LSTM follows at 0.935.</sub></p>

The CNN-LSTM pulls clearly ahead of the other four, particularly at low false positive rates. That low-FPR region is operationally important: if the system raises alerts to a rider or a fleet manager, those alerts need to be trustworthy, not constant background noise.

### D. Precision-Recall Curves

<p align="center">
  <img width="900" height="700" alt="image" src="https://github.com/user-attachments/assets/eae030ba-868c-4a93-b186-bab8abb437a9" />
</p>
<p align="center"><sub><b>Fig. 3.</b> Precision-recall curves for all five models. CNN-LSTM achieves AP = 0.952 and holds precision above 0.9 up to recall ≈ 0.6.</sub></p>

This analysis matters because the dataset carries some class imbalance between normal and rash segments. The CNN-LSTM maintains precision above 0.9 across almost the full recall range up to roughly 0.6 before dropping off, a substantially better curve shape than any of the other four models. The Bi-LSTM (AP = 0.929) is the closest competitor.

### E. Feature Importance

<p align="center">
  <img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/acd5e179-6645-4f90-9fd2-b117af9312b3" />
</p>
<p align="center"><sub><b>Fig. 4.</b> Feature importance from the Random Forest classifier. Lateral acceleration variability (<code>ay_std</code>) and mean forward acceleration (<code>ax_mean</code>) dominate the ranking.</sub></p>

The standout finding is that **`ay_std`**, the standard deviation of lateral acceleration within a window, ranks first, followed by **`ax_mean`**, the mean forward acceleration. Together these two account for close to **9% of total decision weight**.

> 💡 **Takeaway:** Rash riding shows up most clearly as increased variability in side-to-side forces. Swerving, lane-cutting, and unstable braking all produce erratic lateral loads that normal riding simply does not.

Orientation features like azimuth range and rotation energy do appear in the ranking but well below the acceleration statistics, which was somewhat surprising. We had expected heading changes to be a strong signal for swerving maneuvers, but the body dynamics that produce those heading changes appear to show up more clearly in the acceleration channels than in the orientation channels within a fifty-frame window.

---

## 🔬 Experimental Setup

### A. Data Collection

Training data was collected over multiple riding sessions on urban roads around Kolkata. Each session involved a rider going through stretches of normal riding interspersed with deliberately rash maneuvers:

- 🛑 Sudden hard braking
- 🏁 Aggressive acceleration from stops
- ↪️ Sharp turns taken faster than normal
- 〰️ Weaving between lanes

The Android app logged throughout, and the CSV files were annotated afterward, frame by frame, with binary labels: **`0` = normal riding**, **`1` = rash**. Any window that spanned a transition between the two was thrown out rather than assigned an ambiguous label.

### B. Training Details

| Setting | Value |
|:---|:---|
| Window size | 50 frames |
| Stride | 10 frames |
| Train / validation split | 80% / 20% |
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | Adam, learning rate $10^{-3}$ |
| Loss | Binary cross-entropy |

Loss curves stopped meaningfully improving after roughly fifteen epochs for most of the models.

### C. Inference Latency

Response times for the `/predict_window` endpoint were measured on a mid-range development machine with Flask in single-threaded mode. Responses came back in **35 to 80 ms**, covering JSON parsing, NumPy array construction, scaler transform, and the model forward pass. That sits comfortably below the 200 ms polling interval used by the browser frontend, so the visualization never stalls waiting on the server.

---

## ⚠️ Challenges and Limitations

<details>
<summary><b>🕳️ Road Surface Noise</b> (the most persistent practical problem)</summary>

<br/>

Kolkata roads have enough potholes and speed bumps that the vertical acceleration channel picks up a constant layer of high-frequency noise unrelated to rider behavior. In a few sessions the rash probability spiked clearly on a bad stretch of road even though the rider was moving at a perfectly normal pace. We think a **bandpass filter on `az`** before the windowing step would help, but it has not been implemented yet.

</details>

<details>
<summary><b>📡 GPS Gaps</b></summary>

<br/>

GPS dropped out during the handful of sessions run under elevated highways or near large buildings. The system keeps running: the GPS canvas stops updating and the inference pipeline carries on using only the IMU channels. But it means we cannot currently build reliable GPS-fused features or map rash events back to specific road coordinates for those sessions.

</details>

<details>
<summary><b>👤 Generalization Across Devices and Riders</b></summary>

<br/>

Every bit of the training data came from **one device and one rider**. The scaler was fitted on that data, and the model weights encode whatever motion patterns that one rider produces on that one phone. We cannot say with confidence how well any of this transfers to a different phone model, a different mounting position, or a different person's riding style. It almost certainly degrades to some degree, though we have not yet measured how much. Getting a more diverse dataset is the single most important thing we could do to make the system more credible.

</details>

---

## 🚀 Future Work

Roughly in order of how practically impactful we think each direction would be:

- [ ] **📲 On-device inference with TensorFlow Lite.** Right now every prediction needs a server round trip, which works on good Wi-Fi but becomes unreliable on mobile data and fails entirely offline. Moving the model onto the phone removes that dependency and makes the system usable where it matters most: highway stretches and remote roads with poor coverage.
- [ ] **🌍 Expand and diversify the dataset.** Collect data from multiple riders on multiple device models, annotate carefully, and retrain. A **federated learning** approach could let riders contribute without sending raw sensor data to a central server, which matters for privacy.
- [ ] **🏷️ Multi-class event detection.** Moving from binary to classes such as hard braking versus aggressive swerving would make the system far more useful for rider feedback and fleet-level safety analysis.
- [ ] **🗺️ GPS-based heatmapping** of rash episode density onto road maps, turning this from a per-rider tool into something that can identify genuinely dangerous stretches of road.
- [ ] **🎛️ Bandpass filtering of `az`** to reduce road surface noise before windowing.

---

## ✅ Conclusion

We set out to build a rash driving detector for two-wheelers that works without any hardware beyond a phone riders already carry. The resulting system, made of an Android IMU client, a Flask inference server, a trained CNN-LSTM classifier, and a Three.js visualization tool, produces predictions in real time, and the best model reaches **89.4% accuracy with an AUC of 0.955** on the validation data.

The CNN-LSTM's advantage over the other four architectures was consistent across accuracy, AUC, and average precision. Its particular strength was minimizing false negatives, which matters more here than minimizing false alarms. Feature analysis pointed to lateral acceleration variability as the single most informative signal, ahead of orientation features, which shapes our thinking about which preprocessing steps will be most valuable next.

> **Honest caveats:** The training data comes from one rider and one device, and we have not tested how the model holds up in genuinely different conditions. The road noise problem is real and not fully solved. The GPS integration is incomplete. These are the main things standing between this prototype and a version we would call reliable. But the core pipeline works, the latency is acceptable, and the smartphone-only approach is viable. That feels like a solid foundation to build from.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{puitandy2026imu,
  title     = {Inertial Measurement Unit-based Kinematic Anomaly Detection for Two-Wheelers},
  author    = {Vinayak Puitandy, Saif Sahriar, Sanket Manna,
               Vikiron Mondal, Saranya Adhikary, Poojarini Mitra},
  booktitle = {IEEE Conference},
  address   = {Kolkata, India},
  year      = {2026}
}
```

---

## 🔗 References

1. Ministry of Road Transport and Highways, Government of India, "Road Accidents in India, 2022," Transport Research Wing, New Delhi, 2023.
2. G. Engelbrecht, T. Booysen, G. van Rooyen, and F. Bruyns, "Survey of smartphone-based sensing in vehicles for intelligent transportation system applications," *IET Intelligent Transport Systems*, vol. 9, no. 10, pp. 924 to 935, 2015.
3. D. A. Johnson and M. M. Trivedi, "Driving style recognition using a smartphone as a sensor platform," in *Proc. 14th IEEE Int. Conf. Intelligent Transportation Systems (ITSC)*, Washington, DC, USA, 2011, pp. 1609 to 1615.
4. H. Eren, S. Makinist, E. Akin, and A. Yilmaz, "Estimating driving behavior by a smartphone," in *Proc. IEEE Intelligent Vehicles Symposium*, Alcala de Henares, Spain, 2012, pp. 234 to 239.
5. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735 to 1780, 1997.
6. C. Saiprasert, T. Pholprasit, and S. Teeramunkong, "Detection of driving events using sensory data on smartphone," *International Journal of Intelligent Transportation Systems Research*, vol. 12, no. 2, pp. 47 to 57, 2014.
7. A. Sherstinsky, "Fundamentals of recurrent neural network (RNN) and long short-term memory (LSTM) network," *Physica D: Nonlinear Phenomena*, vol. 404, p. 132306, 2020.
8. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *Proc. 12th USENIX Symp. Operating Systems Design and Implementation (OSDI)*, Savannah, GA, USA, 2016, pp. 265 to 283.
9. F. Chollet et al., *Keras*, 2015. [Online]. Available: https://keras.io
10. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825 to 2830, 2011.
11. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785 to 794.

---

<p align="center">
  <sub>Built with 📱 + 🧠 on the roads of Kolkata.</sub>
</p>
