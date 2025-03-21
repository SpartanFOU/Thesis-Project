import numpy as np
import pandas as pd
from pycaret.regression import setup, predict_model,compare_models

class RegressionPyCaret:
    def __init__(self, data, lag=100, forecast_horizon=20, train_size=0.6, validate_size=0.2, test_size=0.2):
        """
        Initializes the TimeSeriesModel with data and parameters.
        
        Args:
        data (list or np.array): Time series data.
        lag (int): Number of past time steps to use as features.
        forecast_horizon (int): Number of time steps ahead to forecast.
        train_size (float): Proportion of data for training.
        validate_size (float): Proportion of data for validation.
        test_size (float): Proportion of data for testing.
        """
        self.data = np.array(data)
        self.lag = lag
        self.forecast_horizon = forecast_horizon
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size
        self.lagged_data = None
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.models = []
        self.results = {}
    
    def create_lagged_features(self):
        """
        Creates lagged features for time series forecasting.
        
        Returns:
        pd.DataFrame: A DataFrame containing lagged features and the target variable.
        """
        X, y = [], []
        for i in range(self.lag, len(self.data) - self.forecast_horizon + 1):
            X.append(self.data[i - self.lag:i])
            y.append(self.data[i + self.forecast_horizon - 1])
        
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=['target'])
        self.lagged_data = pd.concat([X_df, y_df], axis=1)
        return self.lagged_data
    
    def show_head(self, n=5):
        """
        Displays the first few rows of the lagged dataset.
        
        Args:
        n (int): Number of rows to display.
        
        Returns:
        pd.DataFrame: The first `n` rows of the dataset.
        """
        if self.lagged_data is not None:
            return self.lagged_data.head(n)
        else:
            print("Lagged features have not been created yet.")
            return None
    
    def train_validate_test_split(self):
        """
        Splits the data into training, validation, and test sets.
        
        Returns:
        tuple: Three DataFrames for train, validation, and test sets.
        """
        assert np.isclose(self.train_size + self.validate_size + self.test_size, 1.0), "Sizes must sum to 1."
        
        total_size = len(self.lagged_data)
        train_end = int(self.train_size * total_size)
        validate_end = train_end + int(self.validate_size * total_size)
        
        self.data_train = self.lagged_data[:train_end]
        self.data_valid = self.lagged_data[train_end:validate_end]
        self.data_test = self.lagged_data[validate_end:]
        
        return self.data_train, self.data_valid, self.data_test
    
    def setup_and_train_models(self, model_list):
        """
        Sets up the PyCaret regression environment and trains models.
        
        Args:
        model_list (list): List of model identifiers to train.
        
        Returns:
        list: The best models trained.
        """
        if self.data_train is None or self.data_valid is None:
            print("Splitting data before training models...")
            self.train_validate_test_split()
        s = setup(data=self.data_train, test_data=self.data_valid, preprocess=False, imputation_type=None, target='target', session_id=123)
        self.models = compare_models(include=model_list,n_select=len(model_list))
        return self.models
    
    def predict(self):
        """
        Generates predictions using trained models and plots results.
        
        Returns:
        dict: A dictionary containing predictions for each model.
        """
        
        for model in self.models:
            pred = predict_model(model, self.data_test)
            self.results[model] = pred['prediction_label']
        self.results["Original"]=pred['target']   
        return self.results
    def get_trained_models(self):
        """
        Returns the list of trained models.
        """
        return self.models