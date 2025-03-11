import tensorflow as tf
from tensorflow.keras import layers, models

def build_direct_lstm_model(input_seq_len=100, input_dim=1, output_steps=20, lstm_units=64):
    """
    Base Direct LSTM Model: Predicts full future sequence in one step.

    Parameters:
        input_seq_len (int): Length of input sequence (e.g., 100)
        input_dim (int): Number of features per timestep (e.g., 1)
        output_steps (int): Number of future values to predict (e.g., 20)
        lstm_units (int): Number of LSTM units (e.g., 64)

    Returns:
        model (tf.keras.Model): Compiled LSTM model
    """
    inputs = tf.keras.Input(shape=(input_seq_len, input_dim))
    x = layers.LSTM(lstm_units)(inputs)
    outputs = layers.Dense(output_steps)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model