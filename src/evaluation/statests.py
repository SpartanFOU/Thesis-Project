
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
import ruptures as rpt
import seaborn as sns
from scipy.stats import ks_1samp, norm
from scipy.stats import skew
from scipy import stats


def normality_test(x):
    """
    Tests whether the given data follows a normal distribution.
    
    Steps:
    1. Compute and print the skewness of the data.
    2. Perform the Shapiro-Wilk test to check for normality:
       - If p-value < 0.05: Data is not normally distributed.
       - Otherwise: Data is normally distributed.
    3. Perform the Kolmogorov-Smirnov test using a normal distribution as a reference.
    4. Plot a histogram with Kernel Density Estimation (KDE) to visualize distribution.
    5. Create a Q-Q plot to check if data follows a normal distribution.

    Args:
    x (array-like): The dataset to be tested.
    """
    
    print(f"Skewness: {skew(x)}")
    stat, p_value = shapiro(x)
    print(f"Shapiro-Wilk Test Statistic:={stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("Data is not normally distributed")
    else:
        print("Data is normally distributed")
        
    stat, p = ks_1samp(x, norm.cdf)
    print(f"Kolmogorov-Smirnov Test: Statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("Data looks normal (Fail to reject H0)")
    else:
        print("Data is NOT normal (Reject H0)")
        
    sns.histplot(x, kde=True)
    plt.title("Histogram + KDE")
    plt.show()
    stats.probplot(x, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()
    
def adf_test(data):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Steps:
    1. Runs the ADF test on the given dataset.
    2. Prints the test statistic and p-value.
    3. Prints the number of lags used and number of observations.
    4. Provides an interpretation:
       - If p-value < 0.05: The series is likely stationary.
       - Otherwise: The series is likely non-stationary.

    Args:
    data (array-like): Time series data to be tested.
    """
    
    result = adfuller(data)

# Extract results
    adf_statistic = result[0]
    p_value = result[1]
    used_lags = result[2]
    num_observations = result[3]


# Display the results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    print("Used Lags:", used_lags)
    print("Number of Observations:", num_observations)
    
# Explanation of results
    print("\nInterpretation:")
    if p_value < 0.05:
        print(f"The p-value is {p_value:.4f}, which is less than 0.05.")
        print("This means the series is likely stationary.")
    else:
        print(f"The p-value is {p_value:.4f}, which is greater than 0.05.")
        print("This means the series is likely non-stationary.")
        
def acf_test(data, lags=40):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Steps:
    1. ACF plot shows correlation between time series and its past values.
    2. PACF plot helps identify the direct effect of past values on the current value.
    3. Used to determine if the data has autocorrelated patterns.

    Args:
    data (array-like): Time series data.
    lags (int): Number of lags to include in the plots.
    """
    
    plot_acf(data, lags=lags)
    plot_pacf(data, lags=lags)
    plt.show()

def cumsum_plot(data):
    """
    Computes and plots the Cumulative Sum (CUSUM) test to detect shifts in the mean.

    Steps:
    1. Compute the mean of the data.
    2. Compute the cumulative sum of deviations from the mean.
    3. Plot the CUSUM chart to visualize changes over time.
    
    Args:
    data (array-like): Time series data.
    """
    
# Calculate the mean of the time series
    mean_value = np.mean(data)
# Calculate the cumulative sum of deviations from the mean
    cusum = np.cumsum(data - mean_value)
# Plot the CUSUM chart
    plt.figure(figsize=(10, 6))
    plt.plot(cusum, label='CUSUM')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    plt.title('CUSUM (Cumulative Sum) Test')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Sum')
    plt.legend()
    plt.show()

def breakpoints(data):
    """
    Detects structural breaks in a time series using the Bai-Perron Test.

    Steps:
    1. Uses the "l2" least squares model to identify breakpoints.
    2. Applies the PELT (Pruned Exact Linear Time) algorithm to find optimal breakpoints.
    3. Plots the time series with vertical lines indicating detected breakpoints.
    
    Args:
    data (array-like): Time series data.
    """
    
    # Define the model (e.g., "l2" for least squares model)
    model = "l2"

    # Define the number of potential breaks (e.g., 2 breaks)
    penalty = 10  # Regularization parameter
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=penalty)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Time Series")
    for i in result:
     plt.axvline(x=i, color='r', linestyle="--", label=f"Breakpoint at {i}")
    plt.title("Bai-Perron Test: Structural Breaks")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Print detected breakpoints
    print("Detected Breakpoints:", result)

def test_autoregression(data):
    """
    Runs multiple statistical tests to analyze a time series before autoregression modeling.

    Steps:
    1. Normality Test: Checks if data follows a normal distribution.
    2. ADF Test: Checks if the time series is stationary.
    3. ACF & PACF Test: Identifies autocorrelation patterns.
    4. CUSUM Test: Detects changes in mean over time.

    Args:
    data (array-like): Time series data.
    """
    
    normality_test(data)
    adf_test(data)
    acf_test(data)
    cumsum_plot(data)