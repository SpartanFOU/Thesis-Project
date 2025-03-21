from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def build_direct_lstm_model_simple(input_shape, output_steps=20, lstm_units=64):
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
    inputs = Input(shape=input_shape, name="input_layer")
    x = LSTM(lstm_units, name="lstm_layer")(inputs)
    outputs = Dense(output_steps, name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Direct_LSTM_Model_simple")
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
    return model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def build_advanced_direct_lstm(input_shape, output_steps, lstm_units=64, dropout_rate=0.3):
    inputs = Input(shape=input_shape, name="input_layer")

    # Optional Bidirectional Layer (can comment out if not needed)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bidir_lstm_1")(inputs)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = BatchNormalization(name="batch_norm_1")(x)

    # Second LSTM layer
    x = LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)
    x = BatchNormalization(name="batch_norm_2")(x)

    # Dense bottleneck layers
    x = Dense(64, activation='relu', name="dense_1")(x)
    x = Dropout(dropout_rate / 2, name="dropout_3")(x)
    x = Dense(32, activation='relu', name="dense_2")(x)

    # Final output layer (Multi-step direct prediction)
    outputs = Dense(output_steps, name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Advanced_Direct_LSTM_Model")

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae']
    )

    return model



    # MODEL 1: Direct Multi-output LSTM
def build_direct_model_avg(input_shape, output_steps, lstm_units=64):
    """
    Build an optimized Bidirectional LSTM model for direct multi-output prediction.
    """
    inputs = Input(shape=input_shape, name="input_layer")
    
    # First LSTM layer with bidirectional wrapper
    x = Bidirectional(LSTM(2*lstm_units, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(lstm_units, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    outputs = Dense(output_steps)(x)
    
    model = Model(inputs=inputs, outputs=outputs,name="Avarege_Direct_LSTM_Model")
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    
    return model
