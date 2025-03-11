from utils.paths import TMP_DIR
from autoregression.models.DirMultiLSTM import build_direct_lstm_model
from autoregression.models.RecurLSTM import build_recursive_lstm
from autoregression.utils_ar import prepare_direct_lstm_data 
from autoregression.utils_ar import prepare_recursive_lstm_data 
from autoregression.utils_ar import recursive_forecast 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import os
import matplotlib.pyplot as plt
from evaluation import evalresu as er
import importlib

def train_and_save_model(model, X_train, y_train, X_val, y_val,
                         name: str,
                         epochs=30, batch_size=32,
                         patience_early_stopping=10, patience_lr_reduction=5, min_lr=1e-6):
    """
    Train LSTM model and save the best model based on validation loss. 
    Includes EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

    Parameters:
    - model: The model to train.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - name: The name for saving the trained model.
    - epochs: Number of training epochs.
    - batch_size: The batch size for training.
    - patience_early_stopping: Number of epochs to wait before stopping training if no improvement.
    - patience_lr_reduction: Number of epochs to wait before reducing the learning rate.
    - min_lr: The minimum learning rate during learning rate reduction.

    Returns:
    - trained model, training history
    """
    save_path = 'models/saved/' + name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Callback: Save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        save_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    # Callback: Stop training if validation loss does not improve (early stopping)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience_early_stopping, verbose=1, restore_best_weights=True
    )
    
    # Callback: Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience_lr_reduction, 
        min_lr=min_lr, verbose=1
    )
    
    # Fit the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Optionally return the trained model and training history
    return model, history



def plot_training_history(history, title="Training History"):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

data = joblib.load(TMP_DIR / 'output1_smoothed.pkl')
n_past_values=100
n_future_values=20
X_train_dir, y_train_dir,X_val_dir,y_val_dir,X_test_dir,y_test_dir = prepare_direct_lstm_data(data,n_past_values,n_future_values)
dir_model = build_direct_lstm_model(input_seq_len=n_past_values, output_steps=n_future_values,lstm_units=64)
dir_history = train_and_save_model(dir_model,X_train=X_train_dir, y_train=y_train_dir, X_val=X_val_dir, y_val=y_val_dir,name='simple_dir.keras',epochs=50)
plot_training_history(dir_history[1])
dir_pred=dir_model.predict(X_test_dir)
importlib.reload(er)
er.evaluate_n_values(y_test_dir,dir_pred,20)
er.evaluate_full_sequence(y_test_dir,dir_pred,2)



X_train_rec, y_train_rec,X_val_rec,y_val_rec,X_test_rec,y_test_rec = prepare_recursive_lstm_data(data,n_past_values)
rec_model=build_recursive_lstm(n_past_values,64)
rec_history = train_and_save_model(rec_model,X_train=X_train_rec, y_train=y_train_rec, X_val=X_val_rec, y_val=y_val_rec,name='simple_rec.keras',epochs=50)
plot_training_history(rec_history[1])
rec_pred=rec_model.predict(X_test_rec)

y_test_rec[:,0]
importlib.reload(er)
er.evaluate_n_values(y_test_rec,rec_pred,20)
er.evaluate_full_sequence(y_test_rec,rec_pred,0)
n_steps=20
"""
    Perform multi-step forecasting using recursive (autoregressive) strategy.
    
    input_seq: shape (input_len,), last known values
    Returns: predicted sequence of length `n_steps`
"""
input_seq = X_test_rec[0].reshape(1, -1, 1)
predictions = []
import numpy as np
for _ in range(n_steps):
        next_val = rec_model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(next_val)
        # Append next_val and slide window
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)

plt.plot(y_test_rec[0:20])
plt.plot(predictions)

