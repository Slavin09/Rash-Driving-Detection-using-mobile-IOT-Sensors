import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Add,
    Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau






# --- FEATURE ENGINEERING ---

FILE_PATH = '/content/drive/MyDrive/final_rash_data.csv'
WINDOW_SIZE = 50
STEP_SIZE = 25


def engineer_features(df):
    print("--- Step 1: Feature Engineering ---")


    df = df.sort_values(['device_id', 'timestamp']).reset_index(drop=True)

    # Magnitude (Total Force)
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    # Jerk (Change in Force - "Shakiness")
    df['jerk_mag'] = df.groupby('device_id')['acc_mag'].diff().fillna(0)

    # Pitch Delta (Nose diving / Hard Braking)
    df['delta_pitch'] = df.groupby('device_id')['pitch'].diff().fillna(0)

    # Roll Delta (Leaning / Hard Cornering)
    df['delta_roll'] = df.groupby('device_id')['roll'].diff().fillna(0)

    # Azimuth Delta (Turning Rate / "Virtual Gyro")
    df['delta_azimuth'] = df.groupby('device_id')['azimuth'].diff().fillna(0)

    df['delta_azimuth'] = df['delta_azimuth'].apply(
        lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
    )


    df['rotation_energy'] = np.sqrt(df['delta_pitch']**2 + df['delta_roll']**2 + df['delta_azimuth']**2)

    print("Features Created: ax, ay, az, acc_mag, jerk_mag, pitch, roll, delta_pitch, delta_roll, delta_azimuth, rotation_energy")
    return df


def create_statistical_dataset(df):
    print("\n--- Step 2: Feature Selection ---")

    X_stats = []
    y_labels = []

    features_to_analyze = [
        'ax', 'ay', 'az',
        'acc_mag', 'jerk_mag',
        'pitch', 'roll',
        'delta_pitch', 'delta_roll', 'delta_azimuth',
        'rotation_energy'
    ]

    feature_names_out = []

    for device_id, group in df.groupby('device_id'):
        values = group[features_to_analyze].values
        labels = group['is_rash'].values

        for i in range(0, len(group) - WINDOW_SIZE, STEP_SIZE):
            window_data = values[i : i + WINDOW_SIZE]
            window_y = labels[i : i + WINDOW_SIZE]

            label = 1 if np.mean(window_y) > 0.5 else 0

            row = []
            current_names = []

            for f_idx, f_name in enumerate(features_to_analyze):
                f_data = window_data[:, f_idx]

                row.append(np.mean(f_data))
                if i == 0: current_names.append(f"{f_name}_mean")

                row.append(np.std(f_data))
                if i == 0: current_names.append(f"{f_name}_std")

                row.append(np.ptp(f_data))
                if i == 0: current_names.append(f"{f_name}_range")

            X_stats.append(row)
            y_labels.append(label)
            if i == 0: feature_names_out = current_names

    return np.array(X_stats), np.array(y_labels), feature_names_out


try:
    df_raw = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found. Please upload it.")
    exit()


df_engineered = engineer_features(df_raw)

X, y, feat_names = create_statistical_dataset(df_engineered)
print(f"\nDataset Shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Running Permutation Importance...")
result = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()[::-1]

print("\n" + "="*40)
print("TOP FEATURES:")
print("="*40)

top_features = []
for i in range(15):
    idx = sorted_idx[i]
    score = result.importances_mean[idx]
    name = feat_names[idx]
    top_features.append(name)
    print(f"{i+1:2d}. {name:<25} (Score: {score:.4f})")

plt.figure(figsize=(12, 6))

plt.bar(range(15), result.importances_mean[sorted_idx[:15]], align='center')
plt.xticks(range(15), [feat_names[i] for i in sorted_idx[:15]], rotation=45, ha='right')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_selection.png')
plt.show()







# --- MODEL SELECTION ---

indices = np.arange(len(y))
idx_train, idx_test, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)

X_stats_train = X_stats[idx_train]
X_stats_test = X_stats[idx_test]

scaler_stats = StandardScaler()
X_stats_train_sc = scaler_stats.fit_transform(X_stats_train)
X_stats_test_sc = scaler_stats.transform(X_stats_test)

X_seq_train = X_seq[idx_train]
X_seq_test = X_seq[idx_test]

N_train, T, F = X_seq_train.shape
N_test, _, _ = X_seq_test.shape
scaler_seq = StandardScaler()
X_seq_train_sc = scaler_seq.fit_transform(X_seq_train.reshape(-1, F)).reshape(N_train, T, F)
X_seq_test_sc = scaler_seq.transform(X_seq_test.reshape(-1, F)).reshape(N_test, T, F)

print("--- Step 1: Loading & Splitting Data ---")

indices = np.arange(len(y))
idx_train, idx_test, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)


scaler_stats = StandardScaler()
X_stats_train = scaler_stats.fit_transform(X_stats[idx_train])
X_stats_test = scaler_stats.transform(X_stats[idx_test])


scaler_seq = StandardScaler()
N_train, T, F = X_seq[idx_train].shape
N_test, _, _ = X_seq[idx_test].shape
X_seq_train = scaler_seq.fit_transform(X_seq[idx_train].reshape(-1, F)).reshape(N_train, T, F)
X_seq_test = scaler_seq.transform(X_seq[idx_test].reshape(-1, F)).reshape(N_test, T, F)

results = {}


# [1] DECISION TREE -> XGBOOST (The Ultimate Tree)
print("\n[1/5] Training Optimized Tree (XGBoost)...")
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.01, max_depth=6,
    subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1
)
xgb.fit(X_stats_train, y_train)
results['Decision Tree: XGBoost (DT Upgrade)'] = xgb.predict_proba(X_stats_test)[:, 1]

# [2] RANDOM FOREST -> TUNED RF
print("[2/5] Training Optimized Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_stats_train, y_train)
results['Random Forest: Tuned RF'] = rf.predict_proba(X_stats_test)[:, 1]

# [3] MLP -> DEEP RESNET MLP
print("[3/5] Training Optimized MLP (ResNet-style)...")
def build_resnet_mlp():
    inputs = Input(shape=(X_stats_train.shape[1],))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)

    shortcut = x
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Add()([x, shortcut])

    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

mlp = build_resnet_mlp()
mlp.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_stats_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=5)])
results['MLP: ResNet MLP'] = mlp.predict(X_stats_test).flatten()

# [4] LSTM -> BI-DIRECTIONAL LSTM
print("[4/5] Training Optimized LSTM (Bi-Directional)...")
# Bi-LSTM reads sequence Forwards AND Backwards
lstm = Sequential([
    Input(shape=(T, F)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_seq_train, y_train, epochs=40, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=8)])
results['LSTM: Bi-LSTM'] = lstm.predict(X_seq_test).flatten()

# [5] CNN-LSTM -> DEEP ENSEMBLE (3 Models Averaged)
print("[5/5] Training Optimized CNN-LSTM (Ensemble)...")
cnn_preds = np.zeros(len(y_test))
n_ensemble = 3

for i in range(n_ensemble):
    model = Sequential([
        Input(shape=(T, F)),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_seq_train, y_train, epochs=40, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=8)])
    cnn_preds += model.predict(X_seq_test, verbose=0).flatten()

results['CNN-LSTM: Ensemble CNN-LSTM'] = cnn_preds / n_ensemble


print("\n" + "="*60)
print(f"{'MODEL':<25} | {'ACCURACY':<10} | {'AUC SCORE':<10}")
print("="*60)

plt.figure(figsize=(12, 7))


sorted_results = sorted(results.items(), key=lambda x: roc_auc_score(y_test, x[1]), reverse=True)

for name, preds in sorted_results:
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    auc = roc_auc_score(y_test, preds)
    print(f"{name:<25} | {acc:.4f}     | {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title('Final Optimized Model Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('model_comparison_auc.png')
plt.show()

plt.figure(figsize=(9, 7))

for name, preds in results.items():
    precision, recall, _ = precision_recall_curve(y_test, preds)
    ap = average_precision_score(y_test, preds)
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
plt.show()







# --- PRECISION RECALL CURVE ---


fig_cm, axes_cm = plt.subplots(1, len(results), figsize=(20, 4))

for i, (name, preds) in enumerate(results.items()):

    binary_preds = (preds > 0.5).astype(int)
    cm = confusion_matrix(y_test, binary_preds)


    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i], cbar=False)
    axes_cm[i].set_title(f"{name}\nAccuracy: {accuracy_score(y_test, binary_preds):.1%}")
    axes_cm[i].set_xlabel("Predicted")
    if i == 0:
        axes_cm[i].set_ylabel("Actual")
    else:
        axes_cm[i].set_yticks([])

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png')
plt.show()







# --- MODEL EXPORT ---

MODEL_PATH = "rash_driving_model.h5"
SCALER_PATH = "server_scaler.pkl"

print("--- Training Model on 100% Data ---")


N, T, F = X_seq.shape
scaler_final = StandardScaler()
X_flat = X_seq.reshape(-1, F)
X_full_scaled = scaler_final.fit_transform(X_flat).reshape(N, T, F)

print(f"Final Data Shape: {X_full_scaled.shape}")

model = Sequential([
    Input(shape=(T, F)),

    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_full_scaled, y,
    epochs=50,
    batch_size=32,
    verbose=1
)
model.save(MODEL_PATH)
joblib.dump(scaler_final, SCALER_PATH)

print("Saved Model and Scaler")