import mlflow
from mlflow.tracking import MlflowClient
import os

# Set your MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://ard-mlflow.slac.stanford.edu")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print("Registered Models:")


try:
    models = client.search_registered_models()
    
    if not models:
        print("No registered models found!")
        print("\nYou need to register a model in MLflow first.")
    else:
        for model in models:
            print(f"\nModel Name: {model.name}")
            print(f"  Latest Versions:")
            
            # Get all versions of this model
            versions = client.search_model_versions(f"name='{model.name}'")
            
            for version in versions:
                print(f"    Version {version.version}:")
                print(f"      Stage: {version.current_stage}")
                print(f"      Run ID: {version.run_id}")
                print(f"      Status: {version.status}")
                
except Exception as e:
    print(f"Error connecting to MLflow: {str(e)}")
    print("\nMake sure:")
    print("1. MLflow server is running")
    print("2. MLFLOW_TRACKING_URI is set correctly")
