import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler


def load_data(folder_path):
    """
    Loads data from all CSV files in specified folder.

    Args:
        folder_path: The path to the folder containing CSV files.
    
    Returns:
        A Pandas DataFrame containing combined data
    """

    all_files = glob.glob(folder_path + "/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, engine='c', low_memory=False)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)

def preprocess_data(data):
   """
    Preprocesses data for anomaly detection

    Args:
        data: The data to be preprocessed
    
    Returns:
        A Pandas DataFrame containing preprocessed data
   """
   
   # 1. Handle missing values & inf values
   data.replace([np.inf, -np.inf], np.nan, inplace=True)
   data.dropna(inplace=True)

   # 2. Feature selection (Initial Subset)
   features = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Fwd Packet Length Max',  ' Fwd Packet Length Min', ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 'Flow Bytes/s', ' Flow Packets/s']
   data = data[features]

   # 3. Data Transformation
   scaler = MinMaxScaler()
   data[features] = scaler.fit_transform(data[features])

   return data

# Example usage:
folder_path = '/Users/ryangonzalez/anomaly-detection-dashboard/dataset' 
data = load_data(folder_path)

# Test if it works
print(data.head())  # Print the first few rows
print(data.info())  # Print column names and data types