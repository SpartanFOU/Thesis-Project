from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from preprocess import smoothing
from utils.create_input import create_input_deploy
from production.alarm import alarm


def test_whole(lstm_model, real_values, input_window,prediction_window,update_step,mc_simulations,tolerance_lower,tolerance_upper,residual_history,near_pct,off_pct):
    """
    Performs predictive maintenance using an LSTM model with Monte Carlo corrections. The function processes real-time 
    data, applies residual-based Monte Carlo simulations, and raises alarms if predictions approach or exceed tolerance limits.

    Parameters:
    - lstm_model (keras.Model): Pre-trained LSTM model for time-series prediction.
    - real_values (numpy array): Time-series data representing real measured values.
    - input_window (int): Number of past values used for each prediction.
    - prediction_window (int): Future time steps for which predictions are made.
    - update_step (int): Frequency of updating predictions with new real values.
    - mc_simulations (int): Number of Monte Carlo simulations to apply for uncertainty estimation.
    - tolerance_lower (float): Lower tolerance limit for acceptable values.
    - tolerance_upper (float): Upper tolerance limit for acceptable values.
    - residual_history (int): Number of past residuals stored for Monte Carlo sampling.

    Returns:
    - alarm_df (pandas.DataFrame): A DataFrame logging alarm conditions with part numbers and associated risks.

    Process:
    1. **Initialization Phase**:
       - Uses the first 100 known real values to calculate residuals (differences between predictions and actual values).
       - Applies RTS (Rauch-Tung-Striebel) smoothing to remove noise.
       - Predicts 100 future values using the LSTM model and calculates residuals.

    2. **Production Phase**:
       - Iterates over the remaining time series in steps of `update_step`.
       - Updates input data with the latest real values.
       - Applies RTS smoothing on new values.
       - Uses LSTM to predict the next `update_step` values.
       - Computes residuals and updates the residual history.
       - Applies Monte Carlo correction using stored residuals to generate uncertainty bounds.
       - Stores predicted values, confidence intervals, and flags alarms if values approach or exceed tolerance limits.

    3. **Visualization**:
       - Plots the real and predicted values.
       - Displays Monte Carlo confidence intervals.
       - Marks tolerance limits.
       - Adds vertical red dashed lines to indicate alarm-triggered points.

    Alarm System:
    - Calls the `alarm` function to check whether predictions exceed risk thresholds.
    - Stores alarm log entries and visualizes triggered alarms.

    This function provides an automated predictive maintenance system that detects anomalies in real-time data 
    and alerts when values approach or exceed specified limits.
    """
    # Storage for residuals and predictions
    residuals = []
    predicted_parts = []    
    predicted_values = []
    lower_bounds = []
    upper_bounds = []

    print("🔄 Initialization: Predicting 100 future values to create residuals dataset...")

    # Get first 100 known values
    past_values = real_values[: input_window + 99]

    # Apply smoothing to the first 100 values using RTS method
    datasmoothing = smoothing.DataSmoothing(past_values, methods=['RTS'], max_lag=50)
    inp = datasmoothing.get_smoothed_data()['RTS Smoothing']

    # Prepare the sliding window of input data for batch prediction
    X_batch = create_input_deploy(inp, input_len=input_window)  # Get the batch inputs

    # Predict all 100 future values at once
    y_pred = lstm_model.predict(X_batch)[:, -1].flatten()  # Take the last predicted value for each input sequence
    # Actual values corresponding to these predictions
    actual_values = real_values[input_window + prediction_window - 1 : input_window + prediction_window - 1 + 100]


    # Compute residuals
    residuals = actual_values - y_pred

    print("✅ Initialization Complete. Residuals Ready.")
    alarm_log=[]
    update_step = 10  
    for step in range(input_window + 100 -1, len(real_values) -1, update_step):

        # Get 10 new real values
        new_real_values = real_values[step:step + update_step]
        
        # Prepare batch input of shape (10, 100, 1)
        batch_inputs = []
        for i in range(update_step):
            past_values = np.append(past_values, new_real_values[i])[-input_window:]
            past_segment = past_values[-input_window:] # Always take the last 100 values

            
            datasmoothing = smoothing.DataSmoothing(past_segment, methods=['RTS'], max_lag=50)
            smoothed = datasmoothing.get_smoothed_data()['RTS Smoothing']
            
            # Ensure reshaping works correctly
            if smoothed.shape[0] == input_window:
                batch_inputs.append(smoothed.reshape(1, input_window, 1))  # (1,100,1)
            else:
                raise ValueError(f"Expected 100 values, but got {smoothed.shape[0]}")

        lstm_input = np.vstack(batch_inputs)  # Stack to get (10,100,1)

        # Predict 10 future values
        y_pred = lstm_model.predict(lstm_input)[:,-1].flatten()  # Shape (10,)

        # Store and update residuals
        for i in range(update_step):
            predicted_value = y_pred[i]
            actual_value = new_real_values[i]
            new_residual = actual_value - predicted_value
            residuals = np.append(residuals, new_residual)[-residual_history:]  # Keep last 1000 residuals

            # Monte Carlo Correction
            mc_samples = predicted_value + np.random.choice(residuals, size=mc_simulations)
            lower_bound, upper_bound = np.percentile(mc_samples, 0.01), np.percentile(mc_samples, 99.9)

            # Store results for visualization
            predicted_parts.append(step + i+20)
            predicted_values.append(predicted_value)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            alarm_log.append(alarm(mc_samples=mc_samples,part_number=step + i+20,tolerance_lower=tolerance_lower,tolerance_upper=tolerance_upper,near_pct=near_pct,off_pct=off_pct))


    print("✅ Production phase complete. Ready for visualization.")

    # 🔵 3. PLOTTING RESULTS
    plt.figure(figsize=(15, 9))

    # Plot real values
    plt.plot(range(len(real_values)), real_values, label="Real Values", color="blue", alpha=0.6)

    # Plot predicted values
    plt.scatter(predicted_parts, predicted_values, label="Predicted Values", color="red")

    # Plot Monte Carlo confidence region
    plt.fill_between(predicted_parts, lower_bounds, upper_bounds, color="orange", alpha=0.3, label="MC Confidence Region")

    # Plot tolerance limits
    plt.axhline(tolerance_lower, color="black", linestyle="dashed", label="Tolerance Lower Limit",alpha=0.5)
    plt.axhline(tolerance_upper, color="black", linestyle="dashed", label="Tolerance Upper Limit",alpha=0.5)
    alarm_df=pd.DataFrame(alarm_log)
    #alarm_true_list=alarm_df[alarm_df[1]][0].values
    #for i in alarm_true_list:
    #    plt.axvline(i,color='red',linestyle="dashed")
    plt.xlabel("Part Number")
    plt.ylabel("Measurement Value")
    plt.title("Predictive Maintenance - Real vs Predicted Values with MC Confidence Region")
    plt.legend(loc='upper left',frameon=True)
    plt.tight_layout()
    plt.savefig('test with confidence region.pdf',bbox_inches='tight')
    plt.grid()
    plt.show()
    
    return alarm_df