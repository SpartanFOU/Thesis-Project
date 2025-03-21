from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def build_seq2seq_lstm_simple(input_shape, output_steps=20, lstm_units=64):
    """
    Builds and compiles a Seq2Seq LSTM model.
    
    input_shape: Tuple (timesteps, features), e.g., (100, 1)
    output_steps: Number of future time steps to predict
    lstm_units: Number of LSTM units
    """
    # Encoder
    encoder_inputs = Input(shape=input_shape, name="encoder_input")
    encoder_output = LSTM(lstm_units, name="encoder_lstm")(encoder_inputs)
    
    # Repeat context vector for each output timestep
    repeated_context = RepeatVector(output_steps, name="repeat_vector")(encoder_output)

    # Decoder
    decoder_output = LSTM(lstm_units, return_sequences=True, name="decoder_lstm")(repeated_context)

    # Final output layer
    output = TimeDistributed(Dense(1), name="time_distributed_output")(decoder_output)

    # Build model
    model = Model(inputs=encoder_inputs, outputs=output, name="Seq2Seq_LSTM_Model")

    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

    
    return model
