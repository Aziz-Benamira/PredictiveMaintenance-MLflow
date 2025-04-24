import mlflow
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def preprocess_data(input_path, output_path, window_size=5):
    """Preprocess NASA Turbofan dataset, extract features, and create binary label."""
    # Log parameters
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("input_path", input_path)
    
    # Load data
    columns = ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(input_path, sep=r'\s+', header=None, names=columns)
    
    # Feature engineering: rolling statistics
    for sensor in [f'sensor_{i}' for i in range(1, 22)]:
        df[f'{sensor}_mean'] = df.groupby('unit')[sensor].rolling(window=window_size).mean().reset_index(0, drop=True)
        df[f'{sensor}_std'] = df.groupby('unit')[sensor].rolling(window=window_size).std().reset_index(0, drop=True)
    df = df.dropna()
    
    # Create binary label (failure within 30 cycles)
    df['failure'] = df.groupby('unit')['cycle'].transform(lambda x: (x.max() - x) <= 30).astype(int)
    
    # Log metrics
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("num_features", len(df.columns) - 1)
    mlflow.log_metric("failure_ratio", df['failure'].mean())
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Log sample data visualization
    plt.figure(figsize=(10, 6))
    for unit in df['unit'].unique()[:3]:
        unit_data = df[df['unit'] == unit]
        plt.plot(unit_data['cycle'], unit_data['sensor_11'], label=f'Unit {unit}')
    plt.xlabel('Cycle')
    plt.ylabel('Sensor 11 Reading')
    plt.title('Sensor 11 Trends for Sample Units')
    plt.legend()
    plt.savefig("artifacts/sensor_trends.png")
    mlflow.log_artifact("artifacts/sensor_trends.png")
    
    return df

if __name__ == "__main__":
    input_path = "data/raw/train_FD001.txt"
    output_path = "data/processed/train_processed.csv"
    preprocess_data(input_path, output_path)