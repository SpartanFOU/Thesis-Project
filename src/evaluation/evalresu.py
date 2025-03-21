import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MSE, MAE, RMSE, R2, MAPE, SMAPE.
    
    This function computes six common metrics used for model evaluation:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R2 Score
    - Mean Absolute Percentage Error (MAPE)
    - Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Parameters:
    y_true (array-like): The actual target values.
    y_pred (array-like): The predicted target values.
    
    Returns:
    tuple: A tuple containing the values of the six metrics (MSE, MAE, RMSE, R2, MAPE, SMAPE).
    """
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = sqrt(mse)
    
    # R2 Score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    sae=np.sum(np.abs(y_pred - y_true))
    
    return mse, mae, rmse, r2, mape, smape,sae


def plot_results(y_true, y_pred, title="Prediction vs Actual"):
    """
    Plot original and predicted values for comparison.
    
    This function generates a plot that compares the true values (`y_true`) with the predicted values (`y_pred`) over time.
    The actual values are shown in blue and the predicted values in red.

    Parameters:
    y_true (array-like): The actual target values.
    y_pred (array-like): The predicted target values.
    title (str, optional): The title of the plot (default is "Prediction vs Actual").

    Returns:
    None: Displays a plot of actual vs predicted values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", color='blue')
    plt.plot(y_pred, label="Predicted", color='red')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plot residuals (error) to check for patterns.
    
    This function creates a scatter plot of residuals, which is the difference between the true and predicted values.
    The residuals are plotted against the predicted values to help assess the model's performance.

    Parameters:
    y_true (array-like): The actual target values.
    y_pred (array-like): The predicted target values.

    Returns:
    None: Displays a residual plot to evaluate the errors in predictions
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, color='orange')
    plt.hlines(0, min(y_pred), max(y_pred), colors='r', linestyles='dashed')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Error)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
    
    
    
def evaluate_n_values(y_true, y_pred, n:int):
    """
    Evaluate only the 20th predicted value versus the n actual value through the whole dataset.
    
    This function evaluates the n value of the predicted and actual values, calculates the metrics,
    and optionally plots both actual and predicted values of the n step along with the residuals.

    Parameters:
    y_true (array-like): The actual target values.
    y_pred (array-like): The predicted target values.
    n (int): The number of the step to evaluate.

    Returns:
    pd.DataFrame: A DataFrame containing the evaluation metrics for the n value.
    """
    # Get the n_th value from both actual and predicted data

     
        
    mse, mae, rmse, r2, mape, smape,sae= calculate_metrics(y_true, y_pred)
    results = {
        'Metric': ['MSE', 'MAE', 'RMSE', 'R2','MAPE','SMAPE','SAE'],
        'Value': [mse, mae, rmse, r2, mape, smape,sae]
    }
    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    #Plot 20th value
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label=f'Actual {n}_value',  color='blue', linestyle='-', alpha=0.6)
    plt.plot(y_pred, label=f'Predicted {n}_value', color='red', linestyle='--', alpha=0.6)
    plt.title(f"Actual vs Predicted {n}_value")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label='Residuals (Error)', color='orange')
    plt.title("Error Development over Time")
    plt.xlabel('Time ')
    plt.ylabel('Residual (Error)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return results_df

def evaluate_full_sequence(y_true, y_pred, t:int):
    """
    Evaluate the entire predicted sequence (20 steps) against the actual sequence.
    
    This function calculates evaluation metrics for the entire sequence at a given time step `t`, 
    and optionally plots the actual and predicted values of the sequence along with the residuals over time.

    Parameters:
    y_true (array-like): The actual target values.
    y_pred (array-like): The predicted target values.
    t (int): The index of the time step at which the sequence is evaluated.

    Returns:
    pd.DataFrame: A DataFrame containing the evaluation metrics for the sequence at time `t`.
    """
    # Calculate overall error metrics (MSE, MAE, RMSE, R2) for the entire sequence
    mse, mae, rmse, r2, mape, smape,sae = calculate_metrics(y_true[t,:], y_pred[t,:])
    results = {
        'Metric': ['MSE', 'MAE', 'RMSE', 'R2','MAPE','SMAPE','SAE'],
        'Value': [mse, mae, rmse, r2, mape, smape,sae]
    }
    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    # Optionally, plot actual vs predicted over the entire sequence
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[t,:], label='Actual Sequence', color='blue')
    plt.plot(y_pred[t,:], label='Predicted Sequence', color='red', linestyle='dashed')
    plt.title(f"Actual vs Predicted Sequence at t={t}")
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally, plot the error development over time
    residuals = y_true[t,:] - y_pred[t,:]
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label='Residuals (Error)', color='orange')
    plt.title("Error Development over Time")
    plt.xlabel('Time Step')
    plt.ylabel('Residual (Error)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return results_df
