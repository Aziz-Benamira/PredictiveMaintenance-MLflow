# Predictive Maintenance with MLflow

Predictive maintenance pipeline for the NASA Turbofan Jet Engine Dataset using MLflow. Preprocesses sensor data, trains a RandomForestClassifier, evaluates performance, and serves the model as a REST API.

## Project Structure

- `data/raw/`: `train_FD001.txt`, `test_FD001.txt`
- `data/processed/`: `train_processed.csv`, `test_processed.csv`
- `src/`: `preprocess.py`, `train.py`, `preprocess_test.py`, `evaluate.py`, `deploy.py`
- `MLproject`: MLflow configuration
- `conda.yaml`: Dependencies
- `README.md`

## Setup

1. **Install Anaconda**: Download
2. **Clone Repository**:

   ```bash
   git clone https://github.com/Aziz-Benamira/PredictiveMaintenance-MLflow.git
   cd PredictiveMaintenance-MLflow
   ```
3. **Create Environment**:

   ```powershell
   conda env create -f conda.yaml
   conda activate predictive_maintenance_env
   pip install requests
   ```
4. **Start MLflow Server**:

   ```powershell
   cd C:\mlflow_server
   New-Item -Path artifacts -ItemType Directory -Force
   mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
   ```
   - UI: `http://localhost:5000`

## Pipeline

Run in `predictive_maintenance_env` from `C:\Users\benam\Downloads\PredictiveMaintenance-MLflow`:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_CONDA_HOME = "C:\Users\benam\Anaconda3"
```

1. **Preprocess**:

   ```powershell
   mlflow run . -e preprocess --experiment-name PredictiveMaintenance_Experiment
   ```

   - Outputs: `data/processed/train_processed.csv`

2. **Train**:

   ```powershell
   mlflow run . -e train --experiment-name PredictiveMaintenance_Experiment -P n_estimators=100 -P max_depth=5
   ```

   - Outputs: Model `random_forest_model`

3. **Preprocess Test**:

   ```powershell
   mlflow run . -e preprocess_test --experiment-name PredictiveMaintenance_Experiment
   ```

   - Outputs: `data/processed/test_processed.csv`

4. **Evaluate**:

   ```powershell
   mlflow run . -e evaluate --experiment-name PredictiveMaintenance_Experiment -P model_path=runs:/<train_run_id>/random_forest_model
   ```

   - Replace `<train_run_id>` with the `train` run ID from MLflow UI
   - Outputs: `PredictiveMaintenanceModel` version

5. **Deploy (Workaround)**:

   ```powershell
   mlflow models serve -m models:/PredictiveMaintenanceModel/<version> --port 5001 --host 0.0.0.0 --env-manager local
   ```

   - Replace `<version>` with the latest version from MLflow UI
   - Test API:

     ```powershell
     $test_data = @"
     {
         "columns": ["setting_1", "setting_2", "setting_3", "sensor_1", "sensor_1_mean", "sensor_1_std", "sensor_2", "sensor_2_mean", "sensor_2_std", "sensor_3", "sensor_3_mean", "sensor_3_std", "sensor_4", "sensor_4_mean", "sensor_4_std", "sensor_5", "sensor_5_mean", "sensor_5_std", "sensor_6", "sensor_6_mean", "sensor_6_std", "sensor_7", "sensor_7_mean", "sensor_7_std", "sensor_8", "sensor_8_mean", "sensor_8_std", "sensor_9", "sensor_9_mean", "sensor_9_std", "sensor_10", "sensor_10_mean", "sensor_10_std", "sensor_11", "sensor_11_mean", "sensor_11_std", "sensor_12", "sensor_12_mean", "sensor_12_std", "sensor_13", "sensor_13_mean", "sensor_13_std", "sensor_14", "sensor_14_mean", "sensor_14_std", "sensor_15", "sensor_15_mean", "sensor_15_std", "sensor_16", "sensor_16_mean", "sensor_16_std", "sensor_17", "sensor_17_mean", "sensor_17_std", "sensor_18", "sensor_18_mean", "sensor_18_std", "sensor_19", "sensor_19_mean", "sensor_19_std", "sensor_20", "sensor_20_mean", "sensor_20_std", "sensor_21", "sensor_21_mean", "sensor_21_std"],
         "data": [[0.1, 0.2, 100, 518.67, 518.67, 0.5, 642.58, 642.58, 0.4, 1589.7, 1589.7, 0.3, 1400.6, 1400.6, 0.2, 14.62, 14.62, 0.1, 21.61, 21.61, 0.05, 553.75, 553.75, 0.2, 2388.06, 2388.06, 0.3, 9046.19, 9046.19, 0.4, 1.3, 1.3, 0.01, 47.47, 47.47, 0.5, 521.66, 521.66, 0.6, 2388.07, 2388.07, 0.7, 8138.62, 8138.62, 0.8, 8.4195, 8.4195, 0.02, 0.03, 0.03, 0.001, 392, 392, 0.9, 2388, 2388, 0.1, 100, 100, 0, 38.95, 38.95, 0.2, 23.419, 23.419, 0.3]]
     }
     "@
     $test_data | Out-File -FilePath test_data.json
     Invoke-RestMethod -Uri http://localhost:5001/invocations -Method Post -ContentType "application/json" -Body (Get-Content test_data.json -Raw)
     ```
   - Clean up:

     ```powershell
     netstat -a -n -o | find "5001"
     Stop-Process -Id <PID>
     ```

## Troubleshooting

- **Deployment Failure**:

  - If `file://` scheme error occurs, clean up and restart:

    ```powershell
    cd C:\Users\benam\Downloads\PredictiveMaintenance-MLflow
    Remove-Item -Path mlruns -Recurse -Force
    cd C:\mlflow_server
    Remove-Item -Path mlruns.db -Force
    mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
    ```
  - Re-run pipeline.

- **Port Conflicts**:

  ```powershell
  netstat -a -n -o | find "5000"
  Stop-Process -Id <PID>
  ```

## License

MIT License