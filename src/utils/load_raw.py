import pandas as pd
import numpy as np

class DataFrameImporter:
    """
    A class to manage importing and cleaning multiple DataFrames.
    """
    
    def __init__(self):
        self.dataframes = {}  # Store DataFrames in a dictionary

    def load(self, name, file_path, **kwargs):
        """
        Loads a DataFrame and stores it in the dictionary.
        
        Parameters:
            name (str): Identifier for the DataFrame.
            file_path (str): Path to the file (CSV, Excel, etc.).
            kwargs: Additional arguments for pandas read functions.
        """
        if file_path.endswith('.csv'):
            self.dataframes[name] = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx'):
            self.dataframes[name] = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format.")
    
    def get_dataframe(self, name):
        """
        Retrieves a stored DataFrame by name.
        """
        return self.dataframes.get(name, None)
    
    def list_all(self):
        """Creates list of loaded DFs

        Returns:
            list: list of loaded dfs
        """
        return list(self.dataframes.keys()) 
    
    
    
    def combine_all(self, how='inner'):
        """
        Combines all stored DataFrames into one.
        
        Parameters:
            how (str): Merge type ('outer', 'inner', etc.).
        
        Returns:
            pd.DataFrame: The combined DataFrame.
        """
        if not self.dataframes:
            return pd.DataFrame()
        return pd.concat(self.dataframes.values(), axis=0, join=how, ignore_index=True)
    
    def NAN_Cleaning(self, name):
        """
        Cleans a DataFrame by removing rows with errors.
        
        Parameters:
            name (str): The name of the DataFrame to clean.
        """
        if name not in self.dataframes:
            raise ValueError(f"DataFrame '{name}' not found.")
        
        df = self.dataframes[name]
        df = df.replace('', np.nan)
        df = df.replace('#VALUE!', np.nan)
        df = df.dropna()
        #df = df.select_dtypes(include=[np.number])
        df = df.reset_index(drop=True)
        
        self.dataframes[name] = df  # Save the cleaned DataFrame

    def NOK_Cleaning(self, name, border: bool):
        """
        Cleans a DataFrame by removing rows with NOK measures.
        
        Parameters:
            name (str): The name of the DataFrame to clean.
            border (bool): Whether to apply border filtering.
        """
        if name not in self.dataframes:
            raise ValueError(f"DataFrame '{name}' not found.")
        
        df = self.dataframes[name]
        
        if border:
            try:
                df = df.loc[df['RESULT_InclinationBeltDirection__deg_'] > -0.1]
                df = df.loc[df['RESULT_InclinationBeltDirection__deg_'] < 0.9]
                df = df.loc[df['RESULT_Inclination90ToBeltDirection__deg_'] < 0.1]
                df = df.loc[df['RESULT_Inclination90ToBeltDirection__deg_'] > -0.7]
            except KeyError:
                print('DualAxisData')
            
            try:
                df = df.loc[df['Sklon_BD'] > 0]
                df = df.loc[df['Sklon_BD'] < 1]
                df = df.loc[df['Sklon_90_to_BD'] < 0.2]
                df = df.loc[df['Sklon_90_to_BD'] > -1]
            except KeyError:
                print('LineSensorData')
        
        self.dataframes[name] = df  # Save the cleaned DataFrame
