import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from evaluation import evalresu as er
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from datetime import datetime
import os 
from utils.paths import ML_FLOW_DIR


def prepare_direct_lstm_data(series, input_len=100, output_steps=20):
    """
    Prepare data for Direct Multi-Output LSTM with Train/Val/Test split (60/20/20).
    
    Inputs:
        series (np.ndarray): 1D time series
        input_len (int): Length of input window (e.g., 100)
        output_steps (int): Number of future steps to predict (e.g., 20)
        scale (bool): Whether to normalize the data using StandardScaler
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    
    # Create lagged samples
    X, y = [], []
    for t in range(len(series) - input_len - output_steps + 1):
        X.append(series[t : t + input_len])
        y.append(series[t + input_len : t + input_len + output_steps])
    
    X = np.array(X)[..., np.newaxis]  # (samples, input_len, 1)
    y = np.array(y)                   # (samples, output_steps)

    # Split: 60% train, 20% val, 20% test
    n = len(series)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:n-50], y[val_end:n-50]

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_recursive_lstm_data(series, input_len=100):
    """
    Prepare data for Recursive (Autoregressive) LSTM.
    
    Inputs:
        series (np.ndarray): 1D time series
        input_len (int): Length of input sequence
        scale (bool): Whether to normalize the data
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
        (y is next-step value for training autoregressive LSTM)
    """
    X, y = [], []
    for t in range(len(series) - input_len):
        X.append(series[t : t + input_len])
        y.append(series[t + input_len])  # target is next-step value (t+1)

    X = np.array(X)[..., np.newaxis]  # shape: (samples, input_len, 1)
    y = np.array(y).reshape(-1, 1)    # shape: (samples, 1)

    # Split: 60% train, 20% val, 20% test
    n = len(series)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:n-50], y[val_end:n-50]

    return X_train, y_train, X_val, y_val, X_test, y_test

def recursive_forecast(model, input_seq, n_steps=20):
    """
    Perform multi-step forecasting using recursive (autoregressive) strategy.
    
    input_seq: shape (input_len,), last known values
    Returns: predicted sequence of length `n_steps`
    """
    input_seq = input_seq.reshape(1, -1, 1)
    predictions = []

    for _ in range(n_steps):
        next_val = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(next_val)
        
        # Append next_val and slide window
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)

    return np.array(predictions)
def n_from_recursive(model,input_seq,length_of_sequence:int ):
    results=[]
    for _ in range(length_of_sequence):
        input_=input_seq[_]
        result=recursive_forecast(model, input_,)
        results.append(result)
    return results


def prepare_hybrid_dataset(y_pred_direct, y_true, test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepares dataset for refinement LSTM model.
    Input: full predicted sequence (20 steps).
    Target: true 20th value.
    """
    X = y_pred_direct  # shape: (samples, 20)
    y = y_true[:, -1]  # shape: (samples,)

    # Reshape target for LSTM output (samples, 1)
    y = y.reshape(-1, 1)

    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=random_state)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state)

    # Reshape for LSTM input: (samples, time_steps=20, features=1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test



 
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
    
def run_direct_lstm_experiment(
    model_builder_func,
    model_structure_name: str,
    #experiment_name: str,
    run_name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    output_steps: int,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    plot_history_func=True,
    log_additional_metrics=True,
    extra_params: dict = None
):
    """
    Run an LSTM training and evaluation experiment with MLflow tracking.

    Parameters:
    - model_builder_func: function to build the model
    - model_structure_name: registered model name
    - X_train, y_train, X_val, y_val, X_test, y_test: datasets
    - output_steps: number of output steps
    - lstm_units: number of LSTM units
    - batch_size, epochs: training settings
    - plot_history_func : Plot loss history of training 
    - log_additional_metrics: whether to log MSE/MAE/R2 etc.
    - extra_params: dict of additional hyperparameters to log
    """
    input_shape=X_train.shape[1:]
    print(input_shape)
    experiment_name = "Direct_LSTM_Experiments"
    run_name = f"Direct_{model_structure_name}_{lstm_units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Direct_LSTM_Model_{model_structure_name}"
    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(input_shape=input_shape, output_steps=output_steps, lstm_units=lstm_units)
        model.summary()

        history = train_and_save_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            name=f"{registered_model_name}.keras",
            epochs=epochs,
            batch_size=batch_size
        )
        if plot_history_func:
            plot_training_history(history[1])
        # Predictions
        preds = model.predict(X_test)

        # Evaluation
        # adjust import if needed
        results_df = er.evaluate_n_values(y_test[:,-1], preds[:,-1], n=output_steps)
        er.evaluate_full_sequence(y_test, preds, 0)

        # Signature and input example
        input_example = X_test[0:1]
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature
        )

        # Log training metrics
        mlflow.log_metric("val_loss", history[1].history['val_loss'][-1])
        if "val_mae" in history[1].history:
            mlflow.log_metric("val_mae", history[1].history["val_mae"][-1])
        # Log evaluation metrics
        if log_additional_metrics:
            for index, row in results_df.iterrows():
                mlflow.log_metric(row['Metric'], row['Value'])

        # Log hyperparameters
        mlflow.log_params({
            "structure": model_structure_name,
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            **(extra_params or {})
        })
        mlflow.end_run()
        
        
def run_recursive_lstm_experiment(
    model_builder_func,
    model_structure_name: str,
    #experiment_name: str,
    run_name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_predicted_rows: int=100,
    output_steps: int=20,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    plot_history_func=True,
    log_additional_metrics=True,
    extra_params: dict = None
):
    """
    Run an LSTM training and evaluation experiment with MLflow tracking.

    Parameters:
    - model_builder_func: function to build the model
    - model_structure_name: registered model name
    - X_train, y_train, X_val, y_val, X_test, y_test(take y_test_dir): datasets 
    - n_predicted_rows: number of rows to predict,
    - output_steps: number of output steps
    - lstm_units: number of LSTM units
    - batch_size, epochs: training settings
    - plot_history_func : Plot loss history of training 
    - log_additional_metrics: whether to log MSE/MAE/R2 etc.
    - extra_params: dict of additional hyperparameters to log
    """
    input_shape=X_train.shape[1:]
    experiment_name = "Recursive_LSTM_Experiments"
    run_name = f"Recursive_{model_structure_name}_{lstm_units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Recursive_LSTM_Model_{model_structure_name}"
    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(input_shape=input_shape, lstm_units=lstm_units)
        model.summary()

        history = train_and_save_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            name=f"{registered_model_name}.keras",
            epochs=epochs,
            batch_size=batch_size
        )
        if plot_history_func:
            plot_training_history(history[1])
        # Predictions
      
        rec_pred_n=n_from_recursive(model,input_seq=X_test,length_of_sequence=n_predicted_rows)
        rec_pred_n=np.array(rec_pred_n)

        # Evaluation
        # adjust import if needed
        results_df = er.evaluate_n_values(y_test[:n_predicted_rows,-1], rec_pred_n[:,-1], n=20)
        er.evaluate_full_sequence(y_test, rec_pred_n, 0)

        # Signature and input example
        input_example = X_test[0:1]
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature
        )

        # Log training metrics
        mlflow.log_metric("val_loss", history[1].history['val_loss'][-1])
        if "val_mae" in history[1].history:
            mlflow.log_metric("val_mae", history[1].history["val_mae"][-1])
        # Log evaluation metrics
        if log_additional_metrics:
            for index, row in results_df.iterrows():
                mlflow.log_metric(row['Metric'], row['Value'])

        # Log hyperparameters
        mlflow.log_params({
            "structure": model_structure_name,
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            **(extra_params or {})
        })
        mlflow.end_run()  
        
        
        

def run_seq2seq_lstm_experiment(
    model_builder_func,
    model_structure_name: str,
    #experiment_name: str,
    run_name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    output_steps: int,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    plot_history_func=True,
    log_additional_metrics=True,
    extra_params: dict = None
):
    """
    Run an LSTM training and evaluation experiment with MLflow tracking.

    Parameters:
    - model_builder_func: function to build the model
    - model_structure_name: registered model name
    - X_train, y_train, X_val, y_val, X_test, y_test: datasets
    - output_steps: number of output steps
    - lstm_units: number of LSTM units
    - batch_size, epochs: training settings
    - plot_history_func : Plot loss history of training 
    - log_additional_metrics: whether to log MSE/MAE/R2 etc.
    - extra_params: dict of additional hyperparameters to log
    """
    input_shape=X_train.shape[1:]
    experiment_name = "Seq2seq_LSTM_Experiments"
    run_name = f"Seq2seq_{model_structure_name}_{lstm_units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Seq2seq_LSTM_Model_{model_structure_name}"
    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(input_shape=input_shape, output_steps=output_steps, lstm_units=lstm_units)
        model.summary()

        history = train_and_save_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            name=f"{registered_model_name}.keras",
            epochs=epochs,
            batch_size=batch_size
        )
        if plot_history_func:
            plot_training_history(history[1])
        # Predictions
        preds = model.predict(X_test).squeeze(axis=-1)
        # Evaluation
        # adjust import if needed
        results_df = er.evaluate_n_values(y_test[:,-1], preds[:,-1], n=output_steps)
        er.evaluate_full_sequence(y_test, preds, 0)

        # Signature and input example
        input_example = X_test[0:1]
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature
        )

        # Log training metrics
        mlflow.log_metric("val_loss", history[1].history['val_loss'][-1])
        if "val_mae" in history[1].history:
            mlflow.log_metric("val_mae", history[1].history["val_mae"][-1])
        # Log evaluation metrics
        if log_additional_metrics:
            for index, row in results_df.iterrows():
                mlflow.log_metric(row['Metric'], row['Value'])

        # Log hyperparameters
        mlflow.log_params({
            "structure": model_structure_name,
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            **(extra_params or {})
        })
        mlflow.end_run()      
        
def run_direct_refiner_experiment(
    model_builder_func,
    direct_model,
    model_structure_name: str,
    #experiment_name: str,
    run_name: str,
    X_train_dir, y_train_dir, X_val_dir, y_val_dir, X_test_dir, y_test_dir,
    output_steps: int,
    units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    plot_history_func=True,
    log_additional_metrics=True,
    extra_params: dict = None
):
    """
    Run an LSTM training and evaluation experiment with MLflow tracking.

    Parameters:
    - model_builder_func: function to build the model
    - direct_model: trained direct model 
    - model_structure_name: registered model name
    - X_train, y_train, X_val, y_val, X_test, y_test: datasets  for direct model
    - output_steps: number of output steps
    - units: number of LSTM/dense units
    - batch_size, epochs: training settings
    - plot_history_func : Plot loss history of training 
    - log_additional_metrics: whether to log MSE/MAE/R2 etc.
    - extra_params: dict of additional hyperparameters to log
    """
    X_train = direct_model.predict(X_train_dir)
    y_train = y_train_dir[:, -1]
    X_val = direct_model.predict(X_val_dir)
    y_val = y_val_dir[:, -1]
    X_test = direct_model.predict(X_test_dir)
    y_test = y_test_dir[:, -1]
    input_shape=(output_steps,1)
    print(input_shape)
    experiment_name = "Direct_Refiner_Experiments"
    run_name = f"Direct_Refiner{model_structure_name}_{units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Direct_Refiner_Model_{model_structure_name}"
    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(input_shape=input_shape, units=units)
        model.summary()

        history = train_and_save_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            name=f"{registered_model_name}.keras",
            epochs=epochs,
            batch_size=batch_size
        )
        if plot_history_func:
            plot_training_history(history[1])
        # Predictions
        preds = model.predict(X_test).flatten()

        # Evaluation
        # adjust import if needed
        results_df = er.evaluate_n_values(y_test, preds, n=output_steps)

        # Signature and input example
        input_example = X_test[0:1]
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature
        )

        # Log training metrics
        mlflow.log_metric("val_loss", history[1].history['val_loss'][-1])
        if "val_mae" in history[1].history:
            mlflow.log_metric("val_mae", history[1].history["val_mae"][-1])
        # Log evaluation metrics
        if log_additional_metrics:
            for index, row in results_df.iterrows():
                mlflow.log_metric(row['Metric'], row['Value'])

        # Log hyperparameters
        mlflow.log_params({
            "structure": model_structure_name,
            "units": units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            **(extra_params or {})
        })
        mlflow.end_run()    


