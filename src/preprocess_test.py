import mlflow
import pandas as pd
import numpy as np
import os

mlflow.set_tracking_uri("http://localhost:5000")

def preprocess_test_data(input_path, output_path, window_size=5):
    """Preprocess NASA Turbofan test dataset."""
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("input_path", input_path)
    
    columns = ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(input_path, sep=r'\s+', header=None, names=columns)
    
    for sensor in [f'sensor_{i}' for i in range(1, 22)]:
        df[f'{sensor}_mean'] = df.groupby('unit')[sensor].rolling(window=window_size).mean().reset_index(0, drop=True)
        df[f'{sensor}_std'] = df.groupby('unit')[sensor].rolling(window=window_size).std().reset_index(0, drop=True)
    df = df.dropna()
    
    # Placeholder: Assume test labels are provided or derived (for simplicity, use same labeling logic)
    df['failure'] = df.groupby('unit')['cycle'].transform(lambda x: (x.max() - x) <= 30).astype(int)
    
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("num_features", len(df.columns) - 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    mlflow.log_artifact(output_path)
    
    return df

if __name__ == "__main__":
    input_path = "data/raw/test_FD001.txt"
    output_path = "data/processed/test_processed.csv"
    preprocess_test_data(input_path, output_path)