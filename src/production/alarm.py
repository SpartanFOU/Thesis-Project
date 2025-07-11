import numpy as np
def alarm(mc_samples, part_number,tolerance_lower,tolerance_upper,near_pct,off_pct):
    """
    Evaluates the distribution of Monte Carlo simulation samples against tolerance limits and triggers an alarm if 
    too many values are near or exceed these limits.

    Parameters:
    - mc_samples (numpy array): Array of Monte Carlo simulation results.
    - part_number (str): Identifier for the part being evaluated.
    - tolerance_lower (float): Lower tolerance limit for the part.
    - tolerance_upper (float): Upper tolerance limit for the part.
    - near_pct (float): Threshold percentage for values near the tolerance limits to trigger a warning.
    - off_pct (float): Threshold percentage for values outside the tolerance limits to trigger a critical alarm.

    Returns:
    - list: [part_number (str), alarm (bool), cause (str), risk_percentages (dict)]
      - part_number: The identifier of the part.
      - alarm: Boolean indicating whether an alarm is triggered.
      - cause: String indicating the reason for the alarm ("Near" for values close to limits, "OFF" for values out of limits, or empty if no alarm).
      - risk_percentages: Dictionary with percentage distribution of values in different risk categories:
        - "Below Low Limit": Percentage of values below the lower tolerance limit.
        - "Near Low Limit": Percentage of values slightly above the lower tolerance limit.
        - "OK": Percentage of values within the safe range.
        - "Near Upper Limit": Percentage of values slightly below the upper tolerance limit.
        - "Above Upper Limit": Percentage of values exceeding the upper tolerance limit.

    The function prints a warning if too many values are near tolerance limits and a critical alarm if too many values fall outside the limits.
    """
    # Count occurrences in risk regions
    lower_low_limit = tolerance_lower

    upper_high_limit = tolerance_upper 
    tolerance=upper_high_limit-lower_low_limit
    near_low_limit = tolerance_lower+0.05*tolerance
    near_upper_limit = tolerance_upper-0.05*tolerance
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
    if (risk_percentages["Near Low Limit"] + risk_percentages["Near Upper Limit"]) > near_pct:
        print("⚠️ WARNING: Too many values near tolerance limits!")
        alarm=True
        cause="Near"
    elif (risk_percentages["Below Low Limit"] + risk_percentages["Above Upper Limit"]) > off_pct:
        print("❌ CRITICAL: Values out of tolerance! Immediate action required!")
        alarm=True
        cause = "OFF"
    else:
        #print("✅ SAFE: System operating normally.")
        cause=""

    return([part_number,alarm,cause,risk_percentages])
