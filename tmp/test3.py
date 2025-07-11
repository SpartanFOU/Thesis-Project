import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils.paths import TMP_DIR

INPUT_LEN = 100
FUTURE_SHIFT = 20
data = joblib.load(TMP_DIR / 'output1_smoothed_RTS.pkl')
plt.plot(data)
# --- Build features and target ---
X_raw, X_diff1, X_diff20, y_delta = [], [], [], []

for i in range(INPUT_LEN + FUTURE_SHIFT, len(data) - FUTURE_SHIFT):
    raw = data[i - INPUT_LEN:i]
    diff1 = np.diff(data[i - INPUT_LEN - 1:i])  # length = INPUT_LEN
    diff20 = data[i - INPUT_LEN:i] - data[i - INPUT_LEN - FUTURE_SHIFT:i - FUTURE_SHIFT]

    target = data[i + FUTURE_SHIFT] - data[i]  # delta t+20

    X_raw.append(raw)
    X_diff1.append(diff1)
    X_diff20.append(diff20)
    y_delta.append(target)

X_raw = np.array(X_raw)
X_diff1 = np.array(X_diff1)
X_diff20 = np.array(X_diff20)
y_delta = np.array(y_delta).reshape(-1, 1)

# --- Scale features separately ---
scaler_raw = MinMaxScaler()
scaler_diff1 = MinMaxScaler()
scaler_diff20 = MinMaxScaler()
scaler_y = MinMaxScaler()

X_raw_scaled = scaler_raw.fit_transform(X_raw)
X_diff1_scaled = scaler_diff1.fit_transform(X_diff1)
X_diff20_scaled = scaler_diff20.fit_transform(X_diff20)
y_scaled = scaler_y.fit_transform(y_delta)

# --- Stack features along last axis: shape (samples, 100, 3) ---
X_combined = np.stack([X_raw_scaled, X_diff1_scaled, X_diff20_scaled], axis=-1)

# --- Train/test split ---
split_idx = int(len(X_combined) * 0.8)
X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# --- Build model ---
model = Sequential([
    LSTM(64, input_shape=(INPUT_LEN, 3)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# --- Predict and invert scaling ---
y_pred_scaled = model.predict(X_test)
y_pred_delta = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true_delta = scaler_y.inverse_transform(y_test).flatten()

# --- Reconstruct absolute predictions ---
last_known_values = data[split_idx + INPUT_LEN: -FUTURE_SHIFT]
predicted_values = last_known_values + y_pred_delta
actual_values = last_known_values + y_true_delta

# --- Plot results ---
plt.figure(figsize=(10, 5))
plt.plot(actual_values, label="Actual t+20")
plt.plot(y_pred_delta, label="Predicted t+20", alpha=0.7)
plt.title("Predicted vs Actual t+20 Values")
plt.legend()
plt.grid(True)
plt.show()