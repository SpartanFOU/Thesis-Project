import numpy as np

def create_input_deploy(series, input_len=100):
    """
    Prepare X for Direct Multi-Output LSTM 
    
    Inputs:
        series (np.ndarray): 1D time series
        input_len (int): Length of input window (e.g., 100)

    Returns:
        X
    """
    
    # Create lagged samples
    X= []
    for t in range(len(series) - input_len + 1):
        X.append(series[t : t + input_len])

    X = np.array(X)[..., np.newaxis]  # (samples, input_len, 1)

    return X