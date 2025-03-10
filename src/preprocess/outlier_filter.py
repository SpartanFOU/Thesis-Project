import pandas as pd

def filter(df,outputs, score=3):
    """
    Filters out extreme values (outliers) in a DataFrame based on a rolling Z-score method.

    This function applies a moving window approach to compute Z-scores for the specified columns 
    and removes rows where the Z-score exceeds a given threshold.

    Args:
    df (pd.DataFrame): The input DataFrame.
    outputs (list): A list of column names to apply filtering on.
    score (float, optional): The Z-score threshold for filtering. Default is 3.

    Returns:
    pd.DataFrame: A filtered DataFrame with outliers removed.
    """
    
    for output in outputs:  
        window_size=[50,100,50,100,50,100,50,100]
    # Initialize an empty list to store z-scores
        for wind in window_size:
            z_score = pd.Series()  # Initialize an empty Series
            for i in range(0, len(df), wind):  # Start at 0 and step by 100 each time
        #print(f'{i}:{i+99}')
                y = df[output].iloc[i:i+wind]  # Select the correct slice
                y_m = y.mean()  # Mean of the slice
                y_s = y.std()  # Standard deviation of the slice
                z_score_i = ((y - y_m) / y_s).abs()  # Calculate the Z-score
  # Take the absolute value of the Z-score
                z_score = pd.concat([z_score, z_score_i])  # Append the result to z_score
            df = df[z_score < score]
    return(df)



def filter_by_zscore(df, outputs, score=3):
    window_sizes = [50, 100, 50, 100, 50, 100, 50, 100]  # Defined once
    
    for output in outputs:
        z_scores = pd.Series(index=df.index, dtype=float)  # Preallocate Series
        
        for wind in window_sizes:
            for i in range(0, len(df), wind):
                y = df[output].iloc[i:i+wind]
                y_m, y_s = y.mean(), y.std()
                
                if y_s != 0:  # Avoid division by zero
                    z_scores.iloc[i:i+wind] = ((y - y_m) / y_s).abs()
                else:
                    z_scores.iloc[i:i+wind] = 0  # Assign 0 if std is zero
        
            df = df[z_scores < score]
    
    return df