import numpy as np

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
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

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
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

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
def n_from_recursive(length_of_sequence, input_seq,model):
    results=[]
    for _ in range(length_of_sequence):
        input_=input_seq[_]
        result=recursive_forecast(model, input_,)
        results.appeand(result)
        