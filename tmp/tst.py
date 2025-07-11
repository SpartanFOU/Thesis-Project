from utils import load_raw

from matplotlib import pyplot as plt

outputs=['RESULT_InclinationBeltDirection__deg_',
       'RESULT_Inclination90ToBeltDirection__deg_']

importer=load_raw.DataFrameImporter()
importer.load("Data","data/Data Thesis.csv")
#importer.NOK_Cleaning("100k",True)
importer.NAN_Cleaning("Data")
df = importer.get_dataframe("Data")
df.head()
inclination_toBD_original=df[outputs[0]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(inclination_toBD_original.values.reshape(-1, 1))
inclination_toBD_scaled=scaler.transform(inclination_toBD_original.values.reshape(-1, 1)).flatten()
from preprocess import smoothing
import pandas as pd


#methods=['MA', 'Gaussian', 'Wavelet', 'RTS']
datasmoothing = smoothing.DataSmoothing(inclination_toBD_scaled[100:200], methods=['RTS','Gaussian'], max_lag=50)
smoothed=datasmoothing.get_smoothed_data()
smoothed_100=datasmoothing.get_smoothed_data()

denoised_data_test = pd.DataFrame(datasmoothing.compare_smoothing_methods())
print(denoised_data_test)
plt.plot(smoothed['Gaussian Filter'][100:200])
plt.plot(smoothed_100['Gaussian Filter'])
import numpy as np
for i in range(0,1000,100):
    big=smoothed['RTS Smoothing'][i:i+100]
    datasmoothing = smoothing.DataSmoothing(inclination_toBD_scaled[i:i+100], methods=['RTS','Gaussian'], max_lag=50)
    small=datasmoothing.get_smoothed_data()
    plt.plot(big)
    plt.plot(small['RTS Smoothing'])
    plt.show()
    sum=np.sum(small['RTS Smoothing']-big)
    print(sum)
    
 
import mlflow.keras
from utils.paths import ML_FLOW_DIR
from utils.paths import TMP_DIR   
mlflow.set_tracking_uri(ML_FLOW_DIR)

lstm_model = mlflow.keras.load_model("models:/Direct_LSTM_Model_simple/7") 

def prepare_direct_lstm_deploy(series, input_len=100, output_steps=20):
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
    y = np.array(y)  
    return X,y

X,y=prepare_direct_lstm_deploy(smoothed['RTS Smoothing'][300:500],100,20)
y_pred=lstm_model.predict(X)
plt.plot(y[:,-1])
plt.plot(y_pred[:,-1])





datasmoothing = smoothing.DataSmoothing(inclination_toBD_scaled[:220], methods=['RTS'], max_lag=50)
small=datasmoothing.get_smoothed_data()
inp=small['RTS Smoothing']
X_2,y_2=prepare_direct_lstm_deploy(small['RTS Smoothing'],100,20)




def create_input_deploy(series, input_len=100):
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
    X= []
    for t in range(len(series) - input_len + 1):
        X.append(series[t : t + input_len])

    X = np.array(X)[..., np.newaxis]  # (samples, input_len, 1)

    return X

data=inclination_toBD_scaled[:220]
datasmoothing = smoothing.DataSmoothing(inclination_toBD_scaled[:220], methods=['RTS'], max_lag=50)
small=datasmoothing.get_smoothed_data()
inp=small['RTS Smoothing']
y_pred=[]

for i in range(100):
    X_inp=create_input_deploy(inp[i:i+100])
    y_pred_inp=lstm_model.predict(X_inp).flatten()[-1]
    y_pred.append(y_pred_inp)

# Data extraction
data = inclination_toBD_scaled[-340:-120]
data_val=inclination_toBD_scaled[-120:]
datasmoothing = smoothing.DataSmoothing(data, methods=['RTS'], max_lag=50)
inp = datasmoothing.get_smoothed_data()['RTS Smoothing']
n=100
# Prepare batch inputs
X_batch = np.array([create_input_deploy(inp[i:i+100]) for i in range(100)])

# Ensure shape is (n, 100, 1)
X_batch = X_batch.reshape(n, 100, 1)
y_pred = lstm_model.predict(X_batch)[:,-1].flatten()  
res=data[-100:]-y_pred
plt.plot(data[-n:])
plt.plot(y_pred)



plt.plot(res)



plt.plot(inclination_toBD_scaled[15000:25000])
check=inclination_toBD_scaled<=0.266


real_values = inclination_toBD_scaled[15000:23000]

def alarm(mc_samples, part_number):
    # Count occurrences in risk regions
    lower_low_limit = tolerance_lower
    near_low_limit = tolerance_lower*1.1
    near_upper_limit = tolerance_upper*0.9
    upper_high_limit = tolerance_upper 

    risk_counts = {
        "Below Low Limit": np.sum(mc_samples < lower_low_limit),
        "Near Low Limit": np.sum((mc_samples >= lower_low_limit) & (mc_samples < near_low_limit)),
        "OK": np.sum((mc_samples >= near_low_limit) & (mc_samples <= near_upper_limit)),
        "Near Upper Limit": np.sum((mc_samples > near_upper_limit) & (mc_samples <= upper_high_limit)),
        "Above Upper Limit": np.sum(mc_samples > upper_high_limit)
    }

    total_samples = mc_samples.size
    risk_percentages = {key: (value / total_samples) * 100 for key, value in risk_counts.items()}
    alarm=False
    cause=""
    # 🛑 Alarm Logic
    if (risk_percentages["Near Low Limit"] + risk_percentages["Near Upper Limit"]) > 30:
        print("⚠️ WARNING: Too many values near tolerance limits!")
        alarm=True
        cause="Near"
    elif (risk_percentages["Below Low Limit"] + risk_percentages["Above Upper Limit"]) > 1.5:
        print("❌ CRITICAL: Values out of tolerance! Immediate action required!")
        alarm=True
        cause = "OFF"
    else:
        #print("✅ SAFE: System operating normally.")
        cause=""

    return([part_number,alarm,cause,risk_percentages])


input_window = 100  # Past values used for prediction
prediction_window = 20  # LSTM outputs 20 finputure values
update_step = 10  # Every update gives 10 new real values
mc_simulations = 2000  # Monte Carlo iterations
tolerance_lower = 0.26666667
tolerance_upper = 1.46666667
residual_history = 1000  # Keep the last 1000 residuals
def test_whole(model, real_values, input_window,prediction_window,update_step,mc_simulations,tolerance_lower,tolerance_upper,residual_history):
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
            lower_bound, upper_bound = np.percentile(mc_samples, 0), np.percentile(mc_samples, 100)

            # Store results for visualization
            predicted_parts.append(step + i+20)
            predicted_values.append(predicted_value)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            alarm_log.append(alarm(mc_samples=mc_samples,part_number=step + i+20))


    print("✅ Production phase complete. Ready for visualization.")
    print("✅ Production phase complete. Ready for visualization.")
    # 🔵 3. PLOTTING RESULTS
    plt.figure(figsize=(12, 6))

    # Plot real values
    plt.plot(range(len(real_values)), real_values, label="Real Values", color="blue", alpha=0.6)

    # Plot predicted values
    plt.scatter(predicted_parts, predicted_values, label="Predicted Values", color="red")

    # Plot Monte Carlo confidence region
    plt.fill_between(predicted_parts, lower_bounds, upper_bounds, color="orange", alpha=0.3, label="MC Confidence Region")

    # Plot tolerance limits
    plt.axhline(tolerance_lower, color="black", linestyle="dashed", label="Tolerance Lower Limit")
    plt.axhline(tolerance_upper, color="black", linestyle="dashed", label="Tolerance Upper Limit")
    alarm_df=pd.DataFrame(alarm_log)
    alarm_true_list=alarm_df[alarm_df[1]][0].values
    for i in alarm_true_list:
        plt.axvline(i,color='red',linestyle="dashed")
    plt.xlabel("Part Number")
    plt.ylabel("Measurement Value")
    plt.title("Predictive Maintenance - Real vs Predicted Values with MC Confidence Region")
    plt.legend()
    plt.grid()
    plt.show()
    return alarm_df


plt.plot(y_test_pred[-780:],label='test')
plt.plot(predicted_values)
plt.legend()


scaler.transform(np.array((-0.1,0.8)).reshape(-1,1))



alarm_log['2']
























input_window = 100  # Past values used for prediction
prediction_window = 20  # LSTM outputs 20 finputure values
update_step = 10  # Every update gives 10 new real values
mc_simulations = 1000  # Monte Carlo iterations
tolerance_lower = 0.2
tolerance_upper = 0.8
residual_history = 1000  # Keep the last 1000 residuals

# Storage for residuals and predictions
residuals = []
predicted_parts = []
predicted_values = []
lower_bounds = []
upper_bounds = []

print("🔄 Initialization: Predicting 100 future values to create residuals dataset...")

# Get first 100 known values
past_values = real_values[:input_window]

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

update_step = 10  

for step in range(input_window + prediction_window - 1 + 100, len(real_values)-1, update_step):  
    # Get 10 new real values
    new_real_values = real_values[step:step + update_step]

    # Prepare batch input of shape (10, 100, 1)
    batch_inputs = []
    for i in range(update_step):
        past_segment = past_values[-input_window:]  # Always take the last 100 values
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
        residuals = np.append(residuals, new_residual)[-500:]  # Keep last 1000 residuals

        # Monte Carlo Correction
        mc_samples = predicted_value + np.random.choice(residuals, size=mc_simulations)
        lower_bound, upper_bound = np.percentile(mc_samples, 5), np.percentile(mc_samples, 95)

        # Store results for visualization
        predicted_parts.append(step + i)
        predicted_values.append(predicted_value)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        

    # Update past values with new real data
    past_values = np.append(past_values, new_real_values)[-input_window:]  # Keep last 100 values

print("✅ Production phase complete. Ready for visualization.")
print("✅ Production phase complete. Ready for visualization.")
# 🔵 3. PLOTTING RESULTS
plt.figure(figsize=(12, 6))

# Plot real values
plt.plot(range(len(real_values)), real_values, label="Real Values", color="blue", alpha=0.6)

# Plot predicted values
plt.scatter(predicted_parts, predicted_values, label="Predicted Values", color="red")

# Plot Monte Carlo confidence region
plt.fill_between(predicted_parts, lower_bounds, upper_bounds, color="orange", alpha=0.3, label="MC Confidence Region")

# Plot tolerance limits
plt.axhline(tolerance_lower, color="black", linestyle="dashed", label="Tolerance Lower Limit")
plt.axhline(tolerance_upper, color="black", linestyle="dashed", label="Tolerance Upper Limit")

plt.xlabel("Part Number")
plt.ylabel("Measurement Value")
plt.title("Predictive Maintenance - Real vs Predicted Values with MC Confidence Region")
plt.legend()
plt.grid()
plt.show()










def alarm(mc_samples, part_number):
    # Count occurrences in risk regions
    lower_low_limit = tolerance_lower
    near_low_limit = tolerance_lower*1.05
    near_upper_limit = tolerance_upper*0.95
    upper_high_limit = tolerance_upper 

    risk_counts = {
        "Below Low Limit": np.sum(mc_samples < lower_low_limit),
        "Near Low Limit": np.sum((mc_samples >= lower_low_limit) & (mc_samples < near_low_limit)),
        "OK": np.sum((mc_samples >= near_low_limit) & (mc_samples <= near_upper_limit)),
        "Near Upper Limit": np.sum((mc_samples > near_upper_limit) & (mc_samples <= upper_high_limit)),
        "Above Upper Limit": np.sum(mc_samples > upper_high_limit)
    }

    total_samples = mc_samples.size
    risk_percentages = {key: (value / total_samples) * 100 for key, value in risk_counts.items()}
    alarm=False
    cause=""
    # 🛑 Alarm Logic
    if (risk_percentages["Near Low Limit"] + risk_percentages["Near Upper Limit"]) > 50:
        print("⚠️ WARNING: Too many values near tolerance limits!")
        alarm=True
        cause="Near"
    elif (risk_percentages["Below Low Limit"] + risk_percentages["Above Upper Limit"]) > 5:
        print("❌ CRITICAL: Values out of tolerance! Immediate action required!")
        alarm=True
        cause = "OFF"
    else:
        print("✅ SAFE: System operating normally.")

    return([part_number,alarm,cause])
