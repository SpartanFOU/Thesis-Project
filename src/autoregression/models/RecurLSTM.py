from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_recursive_lstm(input_len=100, hidden_units=64):
    """
    Build a basic recursive (autoregressive) LSTM model.
    It predicts the next value (1-step ahead).
    
    Returns: compiled Keras model
    """
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(input_len, 1)))
    model.add(Dense(1))  # Only one output: y_t+1

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
