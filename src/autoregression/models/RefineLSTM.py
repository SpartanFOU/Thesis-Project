from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
import tensorflow as tf


def build_refinement_lstm_model_simple(input_shape, units=64):
    """
    Builds a refinement LSTM model.
    Input shape: (20, 1) — predicted sequence
    Output: single refined 20th value
    """
    inputs = Input(shape=input_shape, name="refinement_input")  # shape=(20, 1)
    x = LSTM(units, name="refinement_lstm")(inputs)
    output = Dense(1, name="refined_20th_output")(x)

    model = Model(inputs=inputs, outputs=output, name="Refinement_LSTM_Model")
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

    return model

def build_refinement_cnn_model_simple(input_shape, num_filters=16, kernel_size=3, units=32):
    """
    Build a simple base CNN model for refinement.
    Takes in a predicted sequence (e.g., from an LSTM model) and outputs a refined single value.

    Parameters:
    - input_steps: Number of input time steps (e.g., 20 predictions from first model).
    - num_filters: Number of filters in the Conv1D layer.
    - kernel_size: Size of convolution kernel.
    - dense_units: Number of units in the Dense layer.

    Returns:
    - Compiled Keras model.
    """
    inputs = Input(shape=input_shape, name="refinement_input")  # shape=(20, 1)

    x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dense(units, activation='relu')(x)
    output = Dense(1, activation='linear', name="refined_output")(x)

    model = Model(inputs, output, name="Base_CNN_Refiner")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def build_refinement_model_lstm_avg(n_input=100, n_output=20, units=64):
    """
    Build a stacked LSTM model to refine the 20th value prediction.
    Uses Reshape layer to properly handle Keras tensors.
    
    Parameters:
    - n_input: Number of past timesteps used as input
    - n_output: Number of future timesteps predicted by the first model
    - units: Base number of units for LSTM layers
    
    Returns:
    - A compiled Keras model for refining the 20th value prediction
    """
    # Input for original sequence
    input_sequence = Input(shape=(n_input, 1), name='original_sequence')
    
    # Input for first model's predictions
    input_predictions = Input(shape=(n_output,), name='first_model_predictions')
    
    # Process the original sequence
    lstm1 = LSTM(units*2, return_sequences=True)(input_sequence)
    lstm1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Process the predictions from the first model using Dense layers
    # This avoids reshape issues with Keras tensors
    pred_dense1 = Dense(units, activation='relu')(input_predictions)
    pred_dense1 = Dropout(0.2)(pred_dense1)
    pred_dense2 = Dense(units//2, activation='relu')(pred_dense1)
    
    # Combine both processed inputs
    combined = Concatenate()([lstm2, pred_dense2])
    
    # Dense layers
    x = Dense(units, activation='relu')(combined)
    x = Dropout(0.1)(x)
    x = Dense(units//2, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer (just one value - the 20th prediction)
    output = Dense(1)(x)
    
    model = Model(inputs=[input_sequence, input_predictions], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    
    return model


def build_average_refinement_lstm(input_shape, units=64, dropout_rate=0.2):
    # Input layer for the refinement model
    refinement_input = Input(shape=input_shape, name="refinement_input")
    
    # First LSTM layer
    x = LSTM(units, return_sequences=True)(refinement_input)
    x = Dropout(dropout_rate)(x)  # Dropout for regularization
    
    # Second LSTM layer
    x = LSTM(units//2, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)  # Dropout for regularization
    
    # Third LSTM layer
    x = LSTM(units//4)(x)
    x = Dropout(dropout_rate)(x)  # Dropout for regularization
    
    # Dense layer for the final refined prediction
    refined_output = Dense(1)(x)
    
    # Model definition
    model = Model(inputs=refinement_input, outputs=refined_output, name='Refiner_LSTM_Average')
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
    
    return model

