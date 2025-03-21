import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
import matplotlib.pyplot as plt
from utils.paths import TMP_DIR
from utils.paths import ML_FLOW_DIR
from autoregression.utils_ar import prepare_direct_lstm_data 
from autoregression.utils_ar import prepare_recursive_lstm_data 
import joblib

data = joblib.load(TMP_DIR / 'output1_smoothed_RTS.pkl')

n_past_values=100
n_future_values=20
input_shape=(n_past_values,1)

X_train_dir, y_train_dir,X_val_dir,y_val_dir,X_test_dir,y_test_dir = prepare_direct_lstm_data(data,n_past_values,n_future_values)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class Seq2SeqLSTM:
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, input_seq_length=100, 
                 target_seq_length=20, dropout_rate=0.2, teacher_forcing_ratio=1.0):
        """
        Seq2Seq LSTM model with teacher forcing and scheduled sampling
        
        Args:
            input_dim: dimensionality of input features
            hidden_dim: hidden dimension of LSTM layers
            output_dim: dimensionality of output features
            input_seq_length: length of input sequences
            target_seq_length: length of target sequences
            dropout_rate: dropout rate for LSTM layers
            teacher_forcing_ratio: probability of using teacher forcing (1.0 = always use teacher forcing)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        self.dropout_rate = dropout_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Build the model
        self.model = self._build_model()
        self.train_model = self._build_train_model()
        
    def _build_model(self):
        """Build the inference model (used for predictions)"""
        # Encoder
        encoder_inputs = Input(shape=(self.input_seq_length, self.input_dim), name='encoder_inputs')
        encoder = LSTM(self.hidden_dim, return_state=True, dropout=self.dropout_rate, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(1, self.output_dim), name='decoder_inputs')
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True, 
                           dropout=self.dropout_rate, name='decoder_lstm')
        decoder_dense = Dense(self.output_dim, activation='linear', name='decoder_dense')
        
        all_outputs = []
        inputs = decoder_inputs
        states = encoder_states
        
        # Decode step by step (autoregressively)
        for i in range(self.target_seq_length):
            # Run decoder on one timestep
            outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
            outputs = decoder_dense(outputs)
            
            # Store prediction
            all_outputs.append(outputs)
            
            # Update states
            states = [state_h, state_c]
            
            # Update input for next timestep (use prediction as next input)
            inputs = outputs
                
        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: tf.concat(x, axis=1))(all_outputs)
        
        # Define the model
        model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        
        return model
    
    def _build_train_model(self):
        """Build the training model (with teacher forcing)"""
        # Encoder
        encoder_inputs = Input(shape=(self.input_seq_length, self.input_dim), name='encoder_inputs')
        encoder = LSTM(self.hidden_dim, return_state=True, dropout=self.dropout_rate, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.target_seq_length, self.output_dim), name='decoder_inputs')
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True, 
                           dropout=self.dropout_rate, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.output_dim, activation='linear', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        
        return model
        
    def compile(self, optimizer='adam', loss='mse'):
        """Compile both models"""
        self.train_model.compile(optimizer=optimizer, loss=loss)
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def predict(self, X):
        """
        Make predictions using the inference model
        
        Args:
            X: input sequences of shape (batch_size, input_seq_length, input_dim)
            
        Returns:
            Predictions of shape (batch_size, target_seq_length, output_dim)
        """
        return self.model.predict(X)
    
    def train_with_teacher_forcing(self, X_train, y_train, X_val=None, y_val=None, 
                                   epochs=50, batch_size=32, callbacks=None):
        """
        Train with full teacher forcing
        
        Args:
            X_train: Training input data (n_samples, input_seq_length, input_dim)
            y_train: Training target data (n_samples, target_seq_length, output_dim)
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Prepare decoder inputs for teacher forcing (shifted targets)
        decoder_input_train = np.zeros((X_train.shape[0], self.target_seq_length, self.output_dim))
        decoder_input_train[:, 0, :] = X_train[:, -1, :]  # First decoder input is last input value
        decoder_input_train[:, 1:, :] = y_train[:, :-1, :]  # Rest is shifted target
        
        validation_data = None
        if X_val is not None and y_val is not None:
            decoder_input_val = np.zeros((X_val.shape[0], self.target_seq_length, self.output_dim))
            decoder_input_val[:, 0, :] = X_val[:, -1, :]
            decoder_input_val[:, 1:, :] = y_val[:, :-1, :]
            validation_data = ([X_val, decoder_input_val], y_val)
        
        history = self.train_model.fit(
            [X_train, decoder_input_train], 
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        return history
    
    def train_with_scheduled_sampling(self, X_train, y_train, X_val=None, y_val=None,
                                     epochs=50, batch_size=32, min_tf_ratio=0.0, 
                                     decay_rate=0.9, callbacks=None):
        """
        Train with scheduled sampling (gradually decreasing teacher forcing)
        
        Args:
            X_train: Training input data (n_samples, input_seq_length, input_dim)
            y_train: Training target data (n_samples, target_seq_length, output_dim)
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size
            min_tf_ratio: Minimum teacher forcing ratio
            decay_rate: Decay rate for teacher forcing ratio
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Training history
        history = {
            'loss': [],
            'val_loss': [] if X_val is not None else None
        }
        
        # Create shared layers for encoder and decoder
        encoder = LSTM(self.hidden_dim, return_state=True, dropout=self.dropout_rate)
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True, 
                           dropout=self.dropout_rate)
        decoder_dense = Dense(self.output_dim, activation='linear')
        
        # Initialize optimizer
        optimizer = tf.keras.optimizers.Adam()
        
        # Number of training samples
        n_train = X_train.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            # Update teacher forcing ratio for this epoch
            current_tf_ratio = max(min_tf_ratio, self.teacher_forcing_ratio * (decay_rate ** epoch))
            print(f"Epoch {epoch+1}/{epochs}, Teacher forcing ratio: {current_tf_ratio:.4f}")
            
            # Shuffle training data
            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train_shuffle = X_train[indices]
            y_train_shuffle = y_train[indices]
            
            epoch_losses = []
            
            # Batch training
            for i in range(0, n_train, batch_size):
                batch_X = X_train_shuffle[i:i+batch_size]
                batch_Y = y_train_shuffle[i:i+batch_size]
                batch_size_actual = batch_X.shape[0]
                
                with tf.GradientTape() as tape:
                    # Run encoder
                    _, state_h, state_c = encoder(batch_X)
                    encoder_states = [state_h, state_c]
                    
                    # Initialize decoder input with the last value from encoder input
                    decoder_input = tf.reshape(batch_X[:, -1, :], (batch_size_actual, 1, self.input_dim))
                    
                    # Initialize decoder states
                    states = encoder_states
                    
                    # Initialize predictions array
                    all_outputs = []
                    
                    # Decode step by step with scheduled sampling
                    for t in range(self.target_seq_length):
                        # Run decoder on one timestep
                        outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=states)
                        outputs = decoder_dense(outputs)
                        
                        # Store prediction
                        all_outputs.append(outputs)
                        
                        # Update states
                        states = [state_h, state_c]
                        
                        # Teacher forcing with scheduled sampling
                        if t < self.target_seq_length - 1:  # No need for the last timestep
                            use_teacher_forcing = tf.random.uniform(()) < current_tf_ratio
                            
                            if use_teacher_forcing:
                                # Use ground truth as next input (teacher forcing)
                                decoder_input = tf.reshape(batch_Y[:, t:t+1, :], 
                                                         (batch_size_actual, 1, self.output_dim))
                            else:
                                # Use prediction as next input
                                decoder_input = outputs
                    
                    # Concatenate all predictions
                    decoder_outputs = tf.concat(all_outputs, axis=1)
                    
                    # Calculate loss
                    loss = tf.reduce_mean(tf.square(decoder_outputs - batch_Y))
                
                # Get gradients and update weights
                variables = [
                    *encoder.trainable_variables,
                    *decoder_lstm.trainable_variables,
                    *decoder_dense.trainable_variables
                ]
                
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                
                epoch_losses.append(loss.numpy())
            
            # Calculate training loss
            train_loss = np.mean(epoch_losses)
            history['loss'].append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                val_loss = np.mean(np.square(val_preds - y_val))
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.6f}")
                
            # Update the model weights for the inference model
            self.model.get_layer('encoder_lstm').set_weights(encoder.get_weights())
            self.model.get_layer('decoder_lstm').set_weights(decoder_lstm.get_weights())
            self.model.get_layer('decoder_dense').set_weights(decoder_dense.get_weights())
            
            # Update the training model weights too
            self.train_model.get_layer('encoder_lstm').set_weights(encoder.get_weights())
            self.train_model.get_layer('decoder_lstm').set_weights(decoder_lstm.get_weights())
            self.train_model.get_layer('decoder_dense').set_weights(decoder_dense.get_weights())
            
            # Call callbacks if provided
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': train_loss})
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Test loss (MSE)
        """
        predictions = self.predict(X_test)
        mse = np.mean(np.square(predictions - y_test))
        return mse

# Example usage
def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Run an experiment with different training strategies
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
    """
    # Get dimensions from data
    input_seq_length = X_train.shape[1]
    target_seq_length = y_train.shape[1]
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[2]
    
    # Create models with different settings
    
    # 1. Full teacher forcing model
    model_tf = Seq2SeqLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        input_seq_length=input_seq_length,
        target_seq_length=target_seq_length,
        teacher_forcing_ratio=1.0
    )
    model_tf.compile(optimizer='adam', loss='mse')
    
    # 2. Scheduled sampling model
    model_ss = Seq2SeqLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        input_seq_length=input_seq_length,
        target_seq_length=target_seq_length,
        teacher_forcing_ratio=1.0
    )
    model_ss.compile(optimizer='adam', loss='mse')
    
    # Train with full teacher forcing
    print("Training with full teacher forcing...")
    history_tf = model_tf.train_with_teacher_forcing(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32
    )
    
    # Train with scheduled sampling
    print("\nTraining with scheduled sampling...")
    history_ss = model_ss.train_with_scheduled_sampling(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32,
        min_tf_ratio=0.0,
        decay_rate=0.9
    )
    
    # Evaluate on test set
    tf_test_loss = model_tf.evaluate(X_test, y_test)
    ss_test_loss = model_ss.evaluate(X_test, y_test)
    
    print("\nTest Results:")
    print(f"Teacher Forcing Test MSE: {tf_test_loss:.6f}")
    print(f"Scheduled Sampling Test MSE: {ss_test_loss:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_tf.history['loss'], label='TF Train')
    plt.plot(history_tf.history['val_loss'], label='TF Val')
    plt.plot(history_ss['loss'], label='SS Train')
    plt.plot(history_ss['val_loss'], label='SS Val')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot example predictions
    plt.subplot(1, 2, 2)
    
    # Get sample predictions
    idx = np.random.randint(0, X_test.shape[0])
    sample_X = X_test[idx:idx+1]
    sample_y = y_test[idx:idx+1]
    
    pred_tf = model_tf.predict(sample_X)
    pred_ss = model_ss.predict(sample_X)
    
    # Plot
    plt.plot(range(input_seq_length), sample_X[0, :, 0], 'b-', label='Input')
    plt.plot(range(input_seq_length, input_seq_length + target_seq_length), 
             sample_y[0, :, 0], 'g-', label='True')
    plt.plot(range(input_seq_length, input_seq_length + target_seq_length), 
             pred_tf[0, :, 0], 'r--', label='TF Pred')
    plt.plot(range(input_seq_length, input_seq_length + target_seq_length), 
             pred_ss[0, :, 0], 'm--', label='SS Pred')
    plt.axvline(x=input_seq_length-1, color='k', linestyle='--')
    plt.title('Prediction Example')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model_tf, model_ss

# Usage example with your existing data
"""
# Assume you already have:
# X_train, y_train, X_val, y_val, X_test, y_test

# Run the experiment
model_tf, model_ss = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test)

# Make predictions on new data
predictions = model_ss.predict(X_new)
"""
y_train_dir=y_train_dir.reshape(y_train_dir.shape[0], y_train_dir.shape[1], 1)
y_val_dir=y_val_dir.reshape(y_val_dir.shape[0], y_val_dir.shape[1], 1)
y_test_dir=y_test_dir.reshape(y_test_dir.shape[0], y_test_dir.shape[1], 1)

input_dim = X_train_dir.shape[2]  # Number of features
output_dim = y_train_dir.shape[2]  # Number of output features

# Create the model
model = Seq2SeqLSTM(
    input_dim=input_dim,
    hidden_dim=64,
    output_dim=output_dim,
    input_seq_length=X_train_dir.shape[1],
    target_seq_length=y_train_dir.shape[1],
    teacher_forcing_ratio=1.0  # Starting ratio
)
model.compile(optimizer='adam', loss='mse')

# Option 1: Train with constant teacher forcing
history_tf = model.train_with_teacher_forcing(
    X_train_dir, y_train_dir, X_val_dir, y_val_dir,
    epochs=30,
    batch_size=32
)

# Option 2: Train with scheduled sampling
history_ss = model.train_with_scheduled_sampling(
    X_train_dir, y_train_dir, X_val_dir, y_val_dir,
    epochs=30,
    batch_size=32,
    min_tf_ratio=0.0,  # Minimum teacher forcing ratio
    decay_rate=0.9     # How quickly to reduce teacher forcing
)
def predict_fixed(model, X):
    """
    Fix for prediction error with the Seq2Seq model
    
    Args:
        model: The Seq2Seq model
        X: Input data with shape (batch_size, input_seq_length, input_dim)
        
    Returns:
        Predictions with shape (batch_size, target_seq_length, output_dim)
    """
    # Get the encoder part of the model
    encoder_inputs = model.model.input
    encoder_lstm = model.model.get_layer('encoder_lstm')
    
    # Get encoder outputs and states
    _, state_h, state_c = encoder_lstm(X)
    states = [state_h, state_c]
    
    # Get the decoder layers
    decoder_lstm = model.model.get_layer('decoder_lstm')
    decoder_dense = model.model.get_layer('decoder_dense')
    
    # Manually create the first decoder input (last value of input sequence)
    decoder_input = X[:, -1:, :]
    
    # Initialize predictions array
    target_seq_length = model.target_seq_length
    batch_size = X.shape[0]
    output_dim = model.output_dim
    all_outputs = []
    
    # Decode step by step
    for _ in range(target_seq_length):
        # Run decoder on one timestep
        outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=states)
        outputs = decoder_dense(outputs)
        
        # Store prediction
        all_outputs.append(outputs)
        
        # Update states
        states = [state_h, state_c]
        
        # Update input for next timestep (use prediction as next input)
        decoder_input = outputs
    
    # Concatenate all predictions
    predictions = tf.concat(all_outputs, axis=1)
    return predictions.numpy()

# Usage with your model:
# predictions = predict_fixed(model, X_test_dir)
predictions = predict_fixed(model,X_test_dir)
from evaluation import evalresu as er
er.evaluate_n_values(y_test_dir.squeeze(axis=-1),predictions.squeeze(axis=-1),20)