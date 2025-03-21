from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def build_recursive_lstm_simple(input_shape, lstm_units=64):
    """
    Build a basic recursive (autoregressive) LSTM model.
    It predicts the next value (1-step ahead).
    
    Returns: compiled Keras model
    """
    inputs = Input(shape=input_shape, name="input_layer")
    x = LSTM(lstm_units, name="lstm_layer")(inputs)
    outputs = Dense(1, name="output_layer")(x)  # Predict one step ahead

    model = Model(inputs=inputs, outputs=outputs, name="Recursive_LSTM_Model_simple")
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

    return model