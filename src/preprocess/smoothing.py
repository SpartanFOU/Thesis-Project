import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
import pywt  # For Wavelet Transform
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
import time

class DataSmoothing:
    def __init__(self, data, methods=None, max_lag=100):
        """
        Initialize the DataSmoothing class.

        Parameters:
        - data: Noisy data to be smoothed.
        - methods: List of smoothing methods to test. If None, all methods are tested.
        - max_lag: Maximum number of lags for autocorrelation analysis.
        """
        self.data = data
        self.methods = methods if methods is not None else ['MA', 'Gaussian', 'Wavelet', 'RTS']
        self.max_lag = max_lag

    def moving_average(self, window_size=5):
        """Apply Moving Average smoothing."""
        start_time = time.perf_counter()
        result = np.convolve(self.data, np.ones(window_size)/window_size, mode='valid')
        end_time = time.perf_counter()
        return {"Smoothed Data": result, "Time": end_time - start_time}

    def gaussian_filter_smooth(self, sigma=1):
        """Apply Gaussian Filter smoothing."""
        start_time = time.perf_counter()
        result = gaussian_filter(self.data, sigma)
        end_time = time.perf_counter()
        return {"Smoothed Data": result, "Time": end_time - start_time}

    def wavelet_transform_smooth(self, wavelet='db1', level=3):
        """Apply Wavelet Transform smoothing."""
        start_time = time.perf_counter()
        coeffs = pywt.wavedec(self.data, wavelet, level=level)
        thresholded_coeffs = [pywt.threshold(c, value=0.2*np.max(c)) for c in coeffs]
        result = pywt.waverec(thresholded_coeffs, wavelet)
        end_time = time.perf_counter()
        return {"Smoothed Data": result, "Time": end_time - start_time}
    
    def kalman_filter_rts_smooth(self, observations, F=np.array([[1]]), H=np.array([[1]]), Q=0.01, R=1, P_initial=1000):
        """Perform Kalman filtering followed by RTS smoothing in one function."""
        start_time = time.perf_counter()
        n_steps = len(observations)
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F, kf.H = F, H
        kf.P = np.eye(1) * P_initial
        kf.R, kf.Q = R, Q

        filtered_state = np.zeros(n_steps)
        P = np.zeros((n_steps, 1, 1))
        P_pred = np.zeros((n_steps, 1, 1))
        x_pred = np.zeros((n_steps, 1))

        for t in range(n_steps):
            kf.predict()
            kf.update(observations[t])
            filtered_state[t] = kf.x[0]
            P[t] = kf.P.copy()
            P_pred[t] = kf.P.copy()
            x_pred[t] = kf.x.copy()

        # Backward pass: RTS Smoother
        for t in range(n_steps - 2, -1, -1):
            S = np.linalg.inv(F @ P[t] @ F.T + P_pred[t])
            K = P[t] @ F.T @ S
            filtered_state[t] += K @ (filtered_state[t + 1] - x_pred[t + 1])
            P[t] += K @ (P[t + 1] - P_pred[t + 1]) @ K.T

        end_time = time.perf_counter()
        return {"Smoothed Data": filtered_state, "Time": end_time - start_time}

    def calculate_correlation(self, smoothed_data):
        """Calculate Pearson correlation between noisy and smoothed data."""
        return pearsonr(self.data[:len(smoothed_data)], smoothed_data)[0]

    def calculate_mse(self, smoothed_data):
        """Calculate Mean Squared Error between noisy and smoothed data."""
        return mean_squared_error(self.data[:len(smoothed_data)], smoothed_data)

    def calculate_autocorrelation(self, smoothed_data):
        """Calculate autocorrelation and find the lag with the highest correlation."""
        noisy_acf = acf(self.data, nlags=self.max_lag, fft=True)
        smoothed_acf = acf(smoothed_data, nlags=self.max_lag, fft=True)
        
        best_lag_noisy = np.argmax(noisy_acf[1:]) + 1
        best_lag_smoothed = np.argmax(smoothed_acf[1:]) + 1

        return {
            "Autocorrelation (Noisy)": noisy_acf[best_lag_noisy],
            "Autocorrelation (Smoothed)": smoothed_acf[best_lag_smoothed],
            "Best Lag (Noisy)": best_lag_noisy,
            "Best Lag (Smoothed)": best_lag_smoothed
        }

    def calculate_deviation(self, smoothed_data):
        """Calculate deviation metrics."""
        original_std = np.std(self.data)
        smoothed_std = np.std(smoothed_data)
        original_var = np.var(self.data)
        smoothed_var = np.var(smoothed_data)
        #mad = np.mean(np.abs(self.data[:len(smoothed_data)] - smoothed_data))
        #max_dev = np.max(np.abs(self.data[:len(smoothed_data)] - smoothed_data))

        return {
            "Standard Deviation Ratio": smoothed_std / original_std,
            "Variance Ratio": smoothed_var / original_var,
            #"Mean Absolute Deviation (MAD)": mad,
            #"Maximum Deviation": max_dev
        }

    def run_tests(self, smoothed_data):
        """Run Correlation, MSE, Autocorrelation, and Deviation analysis."""
        #correlation = self.calculate_correlation(smoothed_data)
        mse = self.calculate_mse(smoothed_data)
        autocorrelation_results = self.calculate_autocorrelation(smoothed_data)
        deviation_results = self.calculate_deviation(smoothed_data)

        return {
            #"Correlation": correlation,
            "MSE": mse,
            **autocorrelation_results,
            **deviation_results
        }

    def compare_smoothing_methods(self):
        """Compare selected smoothing methods."""
        results = {}
        smoothed_data_dict = {}

        if 'MA' in self.methods:
            result = self.moving_average(window_size=5)
            smoothed_data_dict['MA'] = result["Smoothed Data"]
            results['MA'] = self.run_tests(result["Smoothed Data"])
            results['MA']['Time'] = result["Time"]
        
        if 'Gaussian' in self.methods:
            result = self.gaussian_filter_smooth(sigma=3)
            smoothed_data_dict['Gaussian'] = result["Smoothed Data"]
            results['Gaussian'] = self.run_tests(result["Smoothed Data"])
            results['Gaussian']['Time'] = result["Time"]
        
        if 'Wavelet' in self.methods:
            result = self.wavelet_transform_smooth()
            smoothed_data_dict['Wavelet'] = result["Smoothed Data"]
            results['Wavelet'] = self.run_tests(result["Smoothed Data"])
            results['Wavelet']['Time'] = result["Time"]

        if 'RTS' in self.methods:
            result = self.kalman_filter_rts_smooth(self.data)
            smoothed_data_dict['RTS'] = result["Smoothed Data"]
            results['RTS'] = self.run_tests(result["Smoothed Data"])
            results['RTS']['Time'] = result["Time"]

        plt.figure(figsize=(10, 6))
        plt.plot(self.data,'.', label="Noisy Data", alpha=0.7)
        for method, smoothed_data in smoothed_data_dict.items():
            plt.plot(smoothed_data,'.', label=method,alpha=0.5)
        plt.legend()
        plt.title("Comparison of Smoothing Methods")
        plt.show()

        return results
    def get_smoothed_data(self):
        """
        Return smoothed data as a dictionary for use in the main script.

        Returns:
        - A dictionary where keys are method names and values are smoothed data arrays.
        """
        smoothed_data_dict = {"Noisy Data": self.data}

        # Apply selected smoothing methods
        if 'MA' in self.methods:
            smoothed_data_dict['Moving Average'] = self.moving_average(window_size=5)["Smoothed Data"]
        if 'Gaussian' in self.methods:
            smoothed_data_dict['Gaussian Filter'] = self.gaussian_filter_smooth(sigma=3)["Smoothed Data"]
        if 'Wavelet' in self.methods:
            smoothed_data_dict['Wavelet Transform'] = self.wavelet_transform_smooth()["Smoothed Data"]
        if 'RTS' in self.methods:
            smoothed_data_dict['RTS Smoothing'] = self.kalman_filter_rts_smooth(self.data)["Smoothed Data"]

        return smoothed_data_dict

