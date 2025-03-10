
from matplotlib import pyplot as plt

outputs=['RESULT_InclinationBeltDirection__deg_',
       'RESULT_Inclination90ToBeltDirection__deg_']


with open("columns_domain.txt", "r") as file:
    columns = file.read().strip().split(",")

from src import DataFrameImporter
from src import filter
importer=DataFrameImporter()
importer.load("0225","data/test 0225.csv")
#importer.NOK_Cleaning("100k",True)
importer.NAN_Cleaning("0225")
df1 = importer.get_dataframe("0225")
df1=df1[columns]


df1_filtered=filter(df1,outputs)

plt.plot(df1[outputs[0]],label="Original")
plt.plot(df1_filtered[outputs[0]], label="Filtered")
plt.title("Outlier filter")
plt.legend()
plt.show()


from ydata_profiling import ProfileReport
import sweetviz as sv
profile = ProfileReport(df1_filtered, title="Ydata Profiling Report")
profile.to_file("ydata_report_0225.html")
report = sv.analyze(df1_filtered, target_feat=outputs[0])
report.show_html('sweetviz_report_BD_0225.html')

import dtale
import dtale.global_state as global_state
global_state.set_chart_settings({'scatter_points': 200000, '3d_points': 40000})

dtale.show(df1_filtered)

from sklearn.preprocessing import RobustScaler
inclination_toBD_original=df1_filtered[outputs[0]]
scaler = RobustScaler().fit(inclination_toBD_original.values.reshape(-1, 1))

inclination_toBD_original_scaled=scaler.transform(inclination_toBD_original.values.reshape(-1, 1)).flatten()




from src import statests
import importlib
importlib.reload(statests)
statests.test_autoregression(inclination_toBD_original_scaled)
#tests.breakpoints(inclination_toBD_original_scaled)


import seaborn as sns
plt.figure(figsize=(8, 5))
#sns.kdeplot(inclination_toBD_original, label="Original", fill=True, alpha=0.4)
#sns.kdeplot(inclination_toBD_original_scaled, label="Scaled", fill=True, alpha=0.4)
plt.hist(inclination_toBD_original,bins=46,alpha=0.5,label="Original")
plt.hist(inclination_toBD_original_scaled,bins=46,alpha=0.5, label="Scaled")
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Feature Distributions')
plt.legend()
plt.show()




from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
inclination_toBD_original_scaled_transformed_pt=pt.fit_transform(inclination_toBD_original_scaled.reshape(-1,1)).flatten()





from src import smoothing
import pandas as pd



datasmoothing = smoothing.DataSmoothing(inclination_toBD_original.values, methods=['Gaussian','RTS'], max_lag=50)
denoised_data_test = pd.DataFrame(datasmoothing.compare_smoothing_methods())
print(denoised_data_test)




smoothed=datasmoothing.get_smoothed_data()
inclination_toBD_sc_tr_sm=smoothed['Noisy Data']
statests.test_autoregression(inclination_toBD_sc_tr_sm)



inclination_toBD_sc_tr_sm=smoothed['Gaussian Filter']
statests.test_autoregression(inclination_toBD_sc_tr_sm)



inclination_toBD_sc_tr_sm=smoothed['RTS Smoothing']
statests.test_autoregression(inclination_toBD_sc_tr_sm)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Preprocessing the data
# Assuming 'data' is a numpy array with shape (27000,)
data = np.array(inclination_toBD_sc_tr_sm)  # Replace this with your time series data

# Scale the data to [0, 1] range for LSTM to perform bette
def prepare_lstm_data(data, look_back=100, future_steps=20):
    """
    Prepares the time series data for LSTM without scaling.

    Args:
    data (array-like): The raw time series data (e.g., of length 27000).
    look_back (int): The number of previous time steps used as input to predict the next values.
    future_steps (int): The number of future time steps to predict (output).

    Returns:
    X (numpy.ndarray): Input data of shape (samples, look_back) where each sample is a sequence of 'look_back' time steps.
    y (numpy.ndarray): Output data of shape (samples, future_steps) where each sample is a sequence of 'future_steps' values.
    """
    X, y = [], []
    for i in range(len(data) - look_back - future_steps + 1):
        X.append(data[i:(i + look_back)])  # Input: sequence of 'look_back' time steps
        y.append(data[(i + look_back):(i + look_back + future_steps)])  # Output: next 'future_steps' time steps
    return np.array(X), np.array(y)

look_back = 100  # Number of previous time steps to use for prediction
future_steps = 20  # Number of future time steps to predict

X, y = prepare_lstm_data(data, look_back, future_steps)


# Reshape X to be suitable for LSTM input (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 2. Build the LSTM model
model = Sequential()

# Adding Bidirectional LSTM layer (for capturing both directions)
model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(X_train.shape[1], 1)))

# Adding Dropout to reduce overfitting
model.add(Dropout(0.2))

# Adding another LSTM layer for better feature extraction
model.add(LSTM(units=64, return_sequences=False))

# Adding a Dense layer to output the predictions
model.add(Dense(units=future_steps))  # 20 because we are predicting 20 future time steps

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Set up Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

# Train the model with Early Stopping and Learning Rate Scheduling
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, lr_scheduler], verbose=1)

# 4. Evaluate the model


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector

# Hyperparameters
input_steps = 100  # Number of past time steps (input sequence)
future_steps = 20  # Number of future time steps to predict
lstm_units = 128   # LSTM cell size

# ================================
# 1. Define the Encoder
# ================================
encoder_inputs = Input(shape=(input_steps, 1))  # (batch_size, 100, 1)
encoder_lstm = LSTM(lstm_units, return_state=True)  # Only final state matters
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# Save LSTM states (used as context for the decoder)
encoder_states = [state_h, state_c]

# ================================
# 2. Define the Decoder
# ================================
decoder_inputs = Input(shape=(future_steps, 1))  # (batch_size, 20, 1)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)

# Decoder uses the encoder's states as initial states
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Fully connected layer to generate predictions
decoder_dense = Dense(1)  # Predict 1 value per time step
decoder_outputs = decoder_dense(decoder_outputs)

# ================================
# 3. Build and Compile the Model
# ================================
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()

# Example synthetic dataset
X_train = np.random.rand(1000, input_steps, 1)  # (1000 samples, 100 time steps, 1 feature)
Y_train = np.random.rand(1000, future_steps, 1) # (1000 samples, 20 time steps, 1 feature)

# Use shifted Y_train as decoder input
decoder_input_train = np.concatenate([np.zeros((y_train.shape[0], 1, 1)), y_train[:, :-1, :]], axis=1)

# Train the model
model.fit([X_train, decoder_input_train], y_train, epochs=10, batch_size=32)


def predict_future(model, input_seq, future_steps=20):
    input_seq = input_seq.reshape(1, input_steps, 1)  # Reshape for batch
    decoder_input = np.zeros((1, future_steps, 1))  # Start with zeros

    # Predict future steps
    predictions = model.predict([input_seq, decoder_input])
    return predictions.flatten()

# Example prediction
def predict_20th_value(model, X_input):
    # Predict 20 future values, but we only need the 20th one (last value)
    predicted = model.predict(X_input)
    return predicted[:, -1] 
predicted_values = []
actual_values = []
future_predictions = predict_future(model, X_train[0])
print(future_predictions)



for i in range(1000):  # Iterate 1000 times

    predicted_20th = predict_20th_value(model, X[i:i+1])
    
    # Add the predicted value to the list
    predicted_values.append(predicted_20th[0])
    
    # Get the actual value from the test set (corresponding to the next value in the original series)
    actual_values.append(y[i][19])

# 5. Evaluate Performance
# Calculate Mean Squared Error

def predict_full_sequence(model, X_data, future_steps=20):
    """
    Predict future values iteratively for each input in X_data.
    
    Args:
    - model: Trained Seq2Seq model.
    - X_data: Input dataset of shape (num_samples, input_steps, 1).
    - future_steps: Number of future time steps to predict (default=20).
    
    Returns:
    - All predictions as a NumPy array of shape (num_samples, future_steps).
    """
    num_samples = X_data.shape[0]
    predictions = np.zeros((num_samples, future_steps))  # Store all predictions

    for i in range(num_samples):  # Loop through each input sequence
        input_seq = X_data[i].reshape(1, X_data.shape[1], 1)  # Reshape for batch
        decoder_input = np.zeros((1, future_steps, 1))  # Start with zero input

        # Predict future steps for the current input
        predicted_seq = model.predict([input_seq, decoder_input])

        predictions[i] = predicted_seq.flatten()  # Store predictions
    
    return predictions

# Example usage
future_preds = predict_full_sequence(model, X[:1000])  # Predict from X[0] to X[1000]
print(future_preds.shape)
plt.plot(pd.DataFrame(future_preds)[0])
plt.plot(data[:1000])
mse = np.sqrt(mean_squared_error(actual_values, predicted_values))
print(f'Mean Squared Error (MSE): {mse}')






import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Generate dummy data
X, y = prepare_lstm_data(data, look_back, future_steps)


# Reshape X to be suitable for LSTM input (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_samples = 1000
seq_length = 100
n_features = 1

# Define the input layer
input_layer = Input(shape=(seq_length, n_features))

# LSTM layers with regularization (Dropout and L2 regularization)
lstm_layer = LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
lstm_layer = Dropout(0.2)(lstm_layer)  # Dropout for regularization
lstm_layer = LSTM(100)(lstm_layer)

# Separate output units for each time step, using a linear activation for regression
sequence_output = Dense(20, activation='linear')(lstm_layer)  # Separate output for each time step

# Parallel LSTM layer to refine the prediction for the 20th value (same as before)
refine_input = Input(shape=(20, n_features))
refine_lstm = LSTM(50)(refine_input)
refined_output = Dense(n_features, activation='linear')(refine_lstm)

# Combine the models
model = Model(inputs=[input_layer, refine_input], outputs=[sequence_output, refined_output])

# Compile the model with the Adam optimizer and mean squared error loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define learning rate scheduler (ReduceLROnPlateau)
learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Early stopping to prevent overfitting and restore the best weights
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit([X_train, y_train], [y_train, y_train[:, 19, :]], epochs=50, batch_size=32, validation_split=0.2, 
          callbacks=[learning_rate_scheduler, early_stopping])

# Make predictions
sequence_pred, refined_pred = model.predict([X_test, y_test])

# Print model summary
model.summary()

plt.plot(refined_pred)
plt.plot(y_test[:,19,:])


mse = np.sqrt(mean_squared_error(refined_pred, y_test[:,19,:]))






loss, mse = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test MSE: {mse}')

# 5. Predict future values
predicted = model.predict(X_test[0],y_test[0])

plt.plot(predicted)
def predict_20th_value(model, X_input):
    # Predict 20 future values, but we only need the 20th one (last value)
    predicted = model.predict(X_input)
    return predicted[:, -1] 


predicted_values = []
actual_values = []

for i in range(1000):  # Iterate 1000 times

    predicted_20th = predict_20th_value(model, X[i:i+1])
    
    # Add the predicted value to the list
    predicted_values.append(predicted_20th[0])
    
    # Get the actual value from the test set (corresponding to the next value in the original series)
    actual_values.append(y[i][19])

# 5. Evaluate Performance
# Calculate Mean Squared Error


mse = np.sqrt(mean_squared_error(actual_values, predicted_values))
print(f'Mean Squared Error (MSE): {mse}')
predicted_values=pt.inverse_transform(np.array(predicted_values).reshape(-1,1))
predicted_values=scaler.inverse_transform(predicted_values)
# 6. Plot the predicted vs actual values for the first few iterations
plt.figure(figsize=(14, 7))
plt.plot(actual_values, label='Actual Values', color='blue', linestyle='-')
plt.plot(predicted_values, label='Predicted 20th Values', color='red', linestyle='--')
plt.title('Predicted vs Actual (First 50 iterations)')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.show()







data[100:120]
X[0]
y[0]













import numpy as np
import pandas as pd

def create_lagged_features(data, lag=100, forecast_horizon=20):
    """
    Creates lagged features for time series forecasting.

    Args:
    data (list or np.array): Time series data.
    lag (int): Number of past time steps to use for features (e.g., 100).
    forecast_horizon (int): The number of time steps ahead to forecast (e.g., 20 for predicting x120).

    Returns:
    pd.DataFrame: A DataFrame where each row is a set of lagged features and the target value.
    """
    # Convert input data to a pandas DataFrame for easier manipulation
    data = np.array(data)
    # Initialize lists to hold the features and the target variable
    X = []
    y = []
    
    # Loop through the data to create lagged features
    for i in range(lag, len(data) - forecast_horizon + 1):
        # Features: past `lag` observations (e.g., [x1, x2, ..., x100])
        X.append(data[i-lag:i])
        # Target: the forecasted value (e.g., x120)
        y.append(data[i+forecast_horizon-1])
    
    # Convert features and target lists into a DataFrame
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns=['target'])
    
    # Return the final dataset with both features and target
    return pd.concat([X_df, y_df], axis=1)


lag = 100
forecast_horizon = 20

lagged_data = create_lagged_features(data, lag, forecast_horizon)
print(lagged_data.head())
data[119:130]
pd.DataFrame(data)

import pandas as pd
import numpy as np

def train_validate_test_split(data:pd.DataFrame, train_size=0.6, validate_size=0.2, test_size=0.2):
    """
    Splits a time series dataset into training, validation, and test sets and returns them as pandas DataFrames.
    
    Args:
    data (list or np.array): Time series data.
    train_size (float): Proportion of the data to be used for training (default 0.6).
    validate_size (float): Proportion of the data to be used for validation (default 0.2).
    test_size (float): Proportion of the data to be used for testing (default 0.2).
    
    Returns:
    tuple: Three pandas DataFrames corresponding to the train, validation, and test sets.
    """
    # Ensure the sizes sum to 1
    assert np.isclose(train_size + validate_size + test_size, 1.0), "Sizes must sum to 1."

    # Convert the data to a pandas DataFrame
    df=data
    # Calculate the split points
    total_size = len(df)
    train_end = int(train_size * total_size)
    validate_end = train_end + int(validate_size * total_size)

    # Split the data into train, validate, and test DataFrames
    train_df = df[:train_end]
    validate_df = df[train_end:validate_end]
    test_df = df[validate_end:]

    return train_df, validate_df, test_df
data_train, data_valid, data_test=train_validate_test_split(lagged_data)

from pycaret.regression import *
s = setup(data=data_train, test_data=data_valid, preprocess=False, imputation_type=None, target = 'target', session_id = 123)
model=['lr', 'ridge','lightgbm','rf']
best = compare_models(include=model)

for _ in best:
    pred=predict_model(_, data_test)
    plt.plot(pred['target'], label="Original")
    plt.plot(pred['prediction_label'], label="predicted")
    plt.legend()
    plt.show()
models()



















import importlib
importlib.reload(statests)
from src import regression_pycaret as rpc
import importlib
importlib.reload(rpc)
regression=rpc.RegressionPyCaret(data)
regression.create_lagged_features()
regression.setup_and_train_models(['lr', 'ridge', 'lightgbm'])#, 'rf'])
regression_results=regression.predict()
regression_results=pd.DataFrame(regression_results)