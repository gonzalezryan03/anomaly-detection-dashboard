import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler


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
    
    # 1. Handle missing values
   data.dropna(inplace=True)

   # 2. Feature selection (Initial Subset)
   features = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Flow Byts/s', 'Flow Pkts/s']
   data = data[features]

   # 3. Data Transformation
   scaler = StandardScaler()
   data[features] = scaler.fit_transform(data[features])

   return data

# Example usage:
folder_path = '/Users/ryangonzalez/anomaly-detection-dashboard/dataset' 
data = load_data(folder_path)

# Test if it works
print(data.head())  # Print the first few rows
print(data.info())  # Print column names and data types