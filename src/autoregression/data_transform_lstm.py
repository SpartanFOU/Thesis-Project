import numpy as np
from sklearn.model_selection import train_test_split


def prepare_lstm_data(data, look_back=100, future_steps=20):
    """
    Prepares the time series data for LSTM without scaling.

    Args:
    data (array-like): The raw time series data (e.g., of length 27000).
    look_back (int): The number of previous time steps used as input to predict the next values.
    future_steps (int): The number of future time steps to predict (output).

    Returns:
    X (numpy.ndarray): Input data of shape (samples, look_back) where each sample is a sequence of 'look_back' time steps.
    y (numpy.ndarray): Output data of shape (samples, future_steps) where each sample is a sequence of 'future_steps' values.
    """
    X, y = [], []
    for i in range(len(data) - look_back - future_steps + 1):
        X.append(data[i:(i + look_back)])  # Input: sequence of 'look_back' time steps
        y.append(data[(i + look_back):(i + look_back + future_steps)])
    X=np.array(X)
    y=np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    return X_train, X_valid, X_test, y_train, y_valid, y_test 
