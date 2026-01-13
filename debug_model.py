import os
import mlflow
from mlflow.tracking import MlflowClient
from lume_model.models import TorchModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Download artifacts
client = MlflowClient()
model_name = "lcls-cu-inj-model"
model_version = "1"

model_version_obj = client.get_model_version(model_name, model_version)
run_id = model_version_obj.run_id

print(f"Downloading artifacts for run_id: {run_id}")
artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
print(f"Downloaded to: {artifact_path}")

# Find YAML
from pathlib import Path
yaml_files = list(Path(artifact_path).rglob("*.yaml")) + list(Path(artifact_path).rglob("*.yml"))
print(f"\nYAML files found: {yaml_files}")

if yaml_files:
    yaml_path = str(yaml_files[0])
    print(f"\nLoading model from: {yaml_path}")
    
    model = TorchModel(yaml_path)
    
    print(f"\n✓ Model loaded!")
    print(f"Input names: {model.input_names}")
    print(f"Output names: {model.output_names}")
    
    # Check if input_variables exists
    print(f"\nHas input_variables? {hasattr(model, 'input_variables')}")
    if hasattr(model, 'input_variables'):
        print(f"Input variables type: {type(model.input_variables)}")
        print(f"Input variables: {model.input_variables}")
    
    # Check if output_variables exists
    print(f"\nHas output_variables? {hasattr(model, 'output_variables')}")
    if hasattr(model, 'output_variables'):
        print(f"Output variables type: {type(model.output_variables)}")
        print(f"Output variables: {model.output_variables}")
