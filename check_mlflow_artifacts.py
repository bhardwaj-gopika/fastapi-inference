import mlflow
from mlflow.tracking import MlflowClient
import os

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}\n")
print("="*60)
print("Registered Models and Their Artifacts:")
print("="*60)

try:
    models = client.search_registered_models()
    
    if not models:
        print("No registered models found!")
    else:
        for model in models:
            print(f"\n Model: {model.name}")
            
            versions = client.search_model_versions(f"name='{model.name}'")
            
            for version in versions[:3]:  # Show first 3 versions
                print(f"\n  Version {version.version} (Stage: {version.current_stage})")
                print(f"  Run ID: {version.run_id}")
                
                # List artifacts for this run
                try:
                    artifacts = client.list_artifacts(version.run_id)
                    print(f"  Artifacts:")
                    for artifact in artifacts:
                        print(f"    - {artifact.path}")
                        # If it's a directory, show contents
                        if artifact.is_dir:
                            sub_artifacts = client.list_artifacts(version.run_id, artifact.path)
                            for sub in sub_artifacts:
                                print(f"      - {sub.path}")
                except Exception as e:
                    print(f"  Could not list artifacts: {e}")
                    
except Exception as e:
    print(f"Error: {str(e)}")
