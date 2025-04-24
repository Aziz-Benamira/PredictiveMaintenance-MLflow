import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys


mlflow.set_tracking_uri("http://localhost:5000")

def evaluate_model(model_path, test_data_path):
    """Evaluate a model and register it in MLflow."""
    # Load test data
    df = pd.read_csv(test_data_path)
    X_test = df.drop(columns=['failure', 'unit', 'cycle'])
    y_test = df['failure']
    
    # Load model
    model = mlflow.sklearn.load_model(model_path)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    # Register model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    mlflow.register_model(model_uri, "PredictiveMaintenanceModel")
    
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs:/<run_id>/random_forest_model"
    test_data_path = "data/processed/test_processed.csv"
    evaluate_model(model_path, test_data_path)