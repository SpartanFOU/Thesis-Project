import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_variance_data(X_train, y_train, lstm_model,scaler):
    """
    Prepare data for training the variance model.
    
    - X_train: Original input sequence (before LSTM)
    - y_train: True future values (real-world, high variance)
    - lstm_model: Pre-trained LSTM that predicts smoothed values

    Returns:
    - X_var_train: Same as X_train
    - y_var_train: True variance (absolute error between LSTM predictions & real y_train)
    """
    lstm_preds = lstm_model.predict(X_train)
    lstm_preds=scaler.inverse_transform(lstm_preds)
    variance_targets = (y_train - lstm_preds)  # Compute true variance

    return X_train, variance_targets  # X is the same, but Y is now variance


def build_variance_model(input_shape):
    """
    Builds a simple LSTM-based variance predictor.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(20, activation='relu')  # Predict variance for 20 future steps
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the variance model


def create_detailed_alarm_labels(y_real, tolerance_limits, near_threshold=0.05):
    """
    Generate more detailed alarm labels based on tolerance thresholds.

    - y_real: True future values (actual measurements)
    - tolerance_limits: [lower_bound, upper_bound] for acceptable values
    - near_threshold: Percentage margin near tolerance limits (e.g., 5%)

    Returns:
    - alarm_labels: Labels from 0 (Normal) to 4 (Severe Alarm)
    """
    lower_bound, upper_bound = tolerance_limits
    near_lower = lower_bound + (near_threshold * (upper_bound - lower_bound))  # 5% above lower limit
    near_upper = upper_bound - (near_threshold * (upper_bound - lower_bound))  # 5% below upper limit

    alarm_labels = np.zeros(y_real.shape)

    alarm_labels[y_real < lower_bound] = 4  # 🚨 Below Lower Limit
    alarm_labels[(y_real >= lower_bound) & (y_real < near_lower)] = 3  # ⚠️ Near Lower Limit
    alarm_labels[(y_real > upper_bound)] = 2  # 🚨 Above Upper Limit
    alarm_labels[(y_real <= upper_bound) & (y_real > near_upper)] = 1  # ⚠️ Near Upper Limit
    alarm_labels[(y_real >= near_lower) & (y_real <= near_upper)] = 0  # ✅ Normal

    return alarm_labels

def prepare_detailed_classifier_data(X_train, lstm_model, variance_model, y_real, tolerance_limits,scaler):
    """
    Prepare features for the detailed classifier.

    Features:
    - Smoothed predictions (LSTM)
    - Variance predictions
    - Adjusted predictions (LSTM + variance)

    Labels:
    - Alarm categories (0 = Normal, 1 = Near Upper, 2 = Above Upper, 3 = Near Lower, 4 = Below Lower)
    """
    lstm_preds = lstm_model.predict(X_train)
    lstm_preds=scaler.inverse_transform(lstm_preds)
    # Smoothed predictions
    variance_preds = variance_model.predict(X_train)  # Variance
    adjusted_preds = lstm_preds + variance_preds  # Adjusted predictions

    alarm_labels = create_detailed_alarm_labels(y_real[:, -1], tolerance_limits)

    # Combine features for classification
    X_class = np.hstack([lstm_preds, variance_preds, adjusted_preds])
    y_class = alarm_labels  

    return X_class, y_class


from xgboost import XGBClassifier

def build_detailed_alarm_classifier():
    """
    Builds an XGBoost-based alarm classifier for 5 classes.
    """
    model = XGBClassifier(
        n_estimators=150,  
        max_depth=6,  
        learning_rate=0.03,  
        objective="multi:softmax",  
        num_class=5  
    )
    return model

# Train the classifier with detailed labels
tolerance_limits = [-0.1, 0.9]  
X_class_train, y_class_train = prepare_detailed_classifier_data(X_train, lstm_model, variance_model, y_train, tolerance_limits)

detailed_classifier = build_detailed_alarm_classifier()
detailed_classifier.fit(X_class_train, y_class_train)

def detailed_alarm_pipeline(X_input, lstm_model, variance_model, classifier):
    """
    1. Predict 20 future values using LSTM
    2. Estimate variance
    3. Classify into 5 alarm levels
    """
    lstm_preds = lstm_model.predict(X_input)  
    variance_preds = variance_model.predict(X_input)  
    adjusted_preds = lstm_preds + variance_preds  

    alarm_status = classifier.predict(np.hstack([lstm_preds, variance_preds, adjusted_preds]))

    return adjusted_preds, alarm_status




import mlflow.keras
from utils.paths import ML_FLOW_DIR
from utils.paths import TMP_DIR

import joblib

mlflow.set_tracking_uri(ML_FLOW_DIR)

lstm_model = mlflow.keras.load_model("models:/Direct_LSTM_Model_simple/7") 
data_2 = joblib.load(TMP_DIR / 'output1_raw.pkl')
data = joblib.load(TMP_DIR / 'output1_smoothed_RTS.pkl')

from matplotlib import pyplot as plt
plt.plot(data_2)

from autoregression.utils_ar import prepare_direct_lstm_data
X_train_dir, y_train_dir,X_val_dir,y_val_dir,X_test_dir,y_test_dir = prepare_direct_lstm_data(data,100,20)
X_train_dir_raw, y_train_dir_raw,X_val_dir_raw,y_val_dir_raw,X_test_dir_raw,y_test_dir_raw = prepare_direct_lstm_data(data_2,100,20)

import numpy as np
X_test=np.concatenate((X_train_dir,X_val_dir,X_test_dir),axis=0)
X_test_raw=np.concatenate((X_train_dir_raw,X_val_dir_raw,X_test_dir_raw),axis=0)

y_test=np.concatenate((y_train_dir,y_val_dir,y_test_dir),axis=0)
y_test_raw=np.concatenate((y_train_dir_raw,y_val_dir_raw,y_test_dir_raw),axis=0)

y_pred=lstm_model.predict(X_test_dir)
from evaluation import evalresu as er
er.evaluate_n_values(y_test_raw[:,-1],y_pred_scaled[:,-1],20)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(data_2.values.reshape(-1, 1))
y_pred_scaled=scaler.inverse_transform(y_pred)
y_test_scaled=scaler.inverse_transform(y_test)

X_var_train,y_var_train=prepare_variance_data(X_train_dir,y_train_dir_raw,lstm_model,scaler)
variance_model = build_variance_model(X_var_train.shape[1:])
variance_model.fit(X_var_train, y_var_train, epochs=20, batch_size=32, validation_split=0.1)
y_var_pred=variance_model.predict(X_test_dir)
plt.plot(y_pred_scaled[:,-1])
plt.plot(y_var_pred[:,-19])
tolerance_limits = [-0.1, 0.9]  
X_class_train, y_class_train = prepare_detailed_classifier_data(X_train_, lstm_model, variance_model, y_train, tolerance_limits)

detailed_classifier = build_detailed_alarm_classifier()
detailed_classifier.fit(X_class_train, y_class_train)


plt.plot(y_test_dir_raw[:,-1])
plt.plot(scaler.inverse_transform(y_pred[:,-1].reshape(-1,1)).flatten())

import numpy as np

def monte_carlo_correction(predictions, residuals, n=1000,
                           low_limit=-0.1, high_limit=0.9, near_limit_margin=0.05):
    """
    Performs Monte Carlo simulation on LSTM predictions using historical residuals,
    categorizes results into defined risk regions, and returns risk distribution.

    Parameters:
    - predictions: np.array, shape (20,) - LSTM predicted values
    - residuals: np.array, shape (N, 20) - historical residuals dataset
    - n: int, number of Monte Carlo simulations (default: 1000)
    - confidence: float, confidence interval for percentiles (default: 0.9)
    - low_limit, high_limit: float, tolerance thresholds
    - near_limit_margin: float, margin around tolerance limits

    Returns:
    - risk_counts: dict, number of samples in each region
    - risk_percentages: dict, percentage of samples in each region
    """

    # Monte Carlo sampling: add randomly sampled residuals to predictions
    mc_samples = predictions + np.random.choice(residuals.flatten(), size=(n, 20))

    # Define risk region boundaries
    lower_low_limit = low_limit 
    near_low_limit = low_limit + near_limit_margin
    near_upper_limit = high_limit - near_limit_margin
    upper_high_limit = high_limit 

    # Count occurrences in each risk region
    risk_counts = {
        "Below Low Limit": np.sum(mc_samples < lower_low_limit),
        "Near Low Limit": np.sum((mc_samples >= lower_low_limit) & (mc_samples < near_low_limit)),
        "OK": np.sum((mc_samples >= near_low_limit) & (mc_samples <= near_upper_limit)),
        "Near Upper Limit": np.sum((mc_samples > near_upper_limit) & (mc_samples <= upper_high_limit)),
        "Above Upper Limit": np.sum(mc_samples > upper_high_limit)
    }

    # Convert to percentages
    total_samples = mc_samples.size
    risk_percentages = {key: (value / total_samples) * 100 for key, value in risk_counts.items()}

    return risk_counts, risk_percentages


for i in range(10):
    np.shape(X_test_dir[i])
    print(scaler.inverse_transform(lstm_model.predict(X_test_dir[i:i+1])[:,-1].reshape(-1,1)))

y_pred_scaled[:10,-1]
data[100+np.shape(X_train_dir)[0]+np.shape(X_val_dir)[0]:110+np.shape(X_train_dir)[0]+np.shape(X_val_dir)[0]]
y_test_dir[:10,0]



y_pred_ex=lstm_model.predict(X_test_dir[:100])
plt.plot(y_test_dir[:100,-1])
plt.plot(y_test_dir_raw[:100,-1])
plt.plot(scaler.inverse_transform(y_pred_ex[:,19].reshape(-1,1)).flatten())

for i in [0,100,500,1500,1500]:
    plt.plot(y_test_dir_raw[i:i+100,-1])
    plt.plot(scaler.inverse_transform(y_test_dir[i:i+100,-1].reshape(-1,1)).flatten())
    plt.plot(scaler.inverse_transform(y_pred[i:i+100,-1].reshape(-1,1)).flatten())
    plt.show()
    residuals=

for i in range(100):
    y_raw_past=y_test_dir_raw[i:i+100,-1]
    y_pred_past=scaler.inverse_transform(y_pred[i:i+100,-1].reshape(-1,1)).flatten()
    residuals=y_raw_past-y_pred_past
    plt.plot(residuals)
    plt.show()
    plt.hist(residuals,40)
    plt.show()

y_test_dir[:100,-1]
y_pred[:,-1]
np.shape(X_train_dir)[0]


from preprocess.outlier_filter import filter
df_filtered=filter(df,outputs)
inclination_toBD_original=df_filtered[outputs[0]]











import numpy as np
import tensorflow as tf

# Load trained LSTM model
model = tf.keras.models.load_model("your_lstm_model.h5")  # Replace with your model

# Simulated dataset (replace with actual sensor data)
real_values = inclination_toBD_original.values # Replace with real dataset

# Parameters
input_window = 100  # Number of past values used for prediction
prediction_window = 20  # LSTM outputs 20 future values
update_size = 10  # Server update cycle (new values per step)
mc_simulations = 1000  # Monte Carlo iterations
tolerance_lower = -0.1
tolerance_upper = 0.9
near_limit_margin = 0.05  # Defines "near limit" range

# Storage for residuals
residuals = []  

# 🟢 1. INITIALIZATION: Generate 100 known predictions to build residuals dataset
print("🔄 Initialization: Predicting 100 known values to create residuals dataset...")
past_values = real_values[:input_window]  # First 100 known values

for i in range(100):  # Predict one step at a time
    # Prepare input shape (1, 100, 1)
    lstm_input = past_values[-input_window:].reshape(1, input_window, 1)
    
    # Predict next 20 values
    y_pred = model.predict(lstm_input)
    
    # Use only the last predicted value
    predicted_value = y_pred[-1, -1]  
    
    # Store residual (error = actual - predicted)
    actual_value = real_values[input_window + i]
    residuals.append(actual_value - predicted_value)
    
    # Shift past values forward
    past_values = np.append(past_values, actual_value)

# Convert residuals to NumPy array
residuals = np.array(residuals)
print(f"✅ Residuals dataset initialized with {len(residuals)} values.")

# 🟡 2. PRODUCTION PHASE: Continuous updates and risk evaluation
print("🚀 Entering production mode...")
past_values = real_values[:input_window]  # Reset to initial state

for step in range(0, len(real_values) - input_window - prediction_window, update_size):
    print(f"\n🔹 Step {step//update_size + 1}: Predicting next 10 values...")

    # Prepare input (last 100 values)
    lstm_input = past_values[-input_window:].reshape(1, input_window, 1)

    # Predict next 20 values
    y_pred = model.predict(lstm_input)

    # Extract last predicted value for each step
    predicted_values = y_pred[-1, -1]  # Shape (1,)

    # Get actual values from dataset
    actual_values = real_values[input_window + step : input_window + step + update_size]

    # Update residuals
    new_residuals = actual_values - predicted_values[:update_size]
    residuals = np.concatenate([residuals, new_residuals])[-1000:]  # Keep only recent 1000

    # 🛠 Monte Carlo Correction
    mc_samples = predicted_values + np.random.choice(residuals, size=(mc_simulations, update_size))
    
    # Count occurrences in risk regions
    lower_low_limit = tolerance_lower - near_limit_margin
    near_low_limit = tolerance_lower + near_limit_margin
    near_upper_limit = tolerance_upper - near_limit_margin
    upper_high_limit = tolerance_upper + near_limit_margin

    risk_counts = {
        "Below Low Limit": np.sum(mc_samples < lower_low_limit),
        "Near Low Limit": np.sum((mc_samples >= lower_low_limit) & (mc_samples < near_low_limit)),
        "OK": np.sum((mc_samples >= near_low_limit) & (mc_samples <= near_upper_limit)),
        "Near Upper Limit": np.sum((mc_samples > near_upper_limit) & (mc_samples <= upper_high_limit)),
        "Above Upper Limit": np.sum(mc_samples > upper_high_limit)
    }

    total_samples = mc_samples.size
    risk_percentages = {key: (value / total_samples) * 100 for key, value in risk_counts.items()}

    # 🛑 Alarm Logic
    if (risk_percentages["Near Low Limit"] + risk_percentages["Near Upper Limit"]) > 50:
        print("⚠️ WARNING: Too many values near tolerance limits!")
    elif (risk_percentages["Below Low Limit"] + risk_percentages["Above Upper Limit"]) > 5:
        print("❌ CRITICAL: Values out of tolerance! Immediate action required!")
    else:
        print("✅ SAFE: System operating normally.")

    # Update past values for next iteration
    past_values = np.append(past_values, actual_values)
