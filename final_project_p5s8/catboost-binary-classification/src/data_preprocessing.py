import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(syndata_path):
    """Load the synthetic dataset from the specified path."""
    return pd.read_csv(syndata_path)

def standardize_numerical(data):
    """Apply Z-standardization to all numerical variables in the dataset."""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

def prepare_data(syndata_path):
    """Load and preprocess the data for modeling."""
    data = load_data(syndata_path)
    data = standardize_numerical(data)
    return data