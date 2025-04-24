import mlflow
import subprocess
import sys
import time
import requests
import os

def deploy_model(model_name, version, port=5001):
    """Deploy a registered model as a local REST API"""
    # Set tracking URI (make configurable)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Start a run in the specified experiment
    experiment = mlflow.get_experiment_by_name("PredictiveMaintenance_Experiment")
    if not experiment:
        mlflow.create_experiment("PredictiveMaintenance_Experiment")
    
    with mlflow.start_run(experiment_id=experiment.experiment_id if experiment else None):
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "model_version": version,
            "model_port": port,
            "deployment_type": "local"
        })
        
        # Model URI
        model_uri = f"models:/{model_name}/{version}"
        
        # Verify model exists
        try:
            client = mlflow.client.MlflowClient()
            client.get_model_version(model_name, version)
        except Exception as e:
            mlflow.log_metric("deployment_status", 0)
            raise Exception(f"Model {model_uri} not found: {str(e)}")
        
        # Serve model
        cmd = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--env-manager", "local"
        ]
        
        with open("server_log.txt", "w") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
        
        # Wait for server to start
        time.sleep(30)
        
        # Check if server is running
        try:
            response = requests.get(f"http://localhost:{port}/ping", timeout=10)
            if response.status_code == 200:
                mlflow.log_metric("deployment_status", 1)
                mlflow.log_artifact("server_log.txt")
                print(f"Model deployed successfully at http://localhost:{port}")
                return process
            else:
                raise Exception(f"Server returned status {response.status_code}")
        except Exception as e:
            mlflow.log_metric("deployment_status", 0)
            with open("server_log.txt", "r") as log_file:
                stderr = log_file.read()
            print(f"Deployment failed: {str(e)}\nServer log: {stderr}")
            mlflow.log_artifact("server_log.txt")
            process.terminate()
            raise

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "PredictiveMaintenanceModel"
    version = sys.argv[2] if len(sys.argv) > 2 else "1"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5001
    deploy_model(model_name, version, port)