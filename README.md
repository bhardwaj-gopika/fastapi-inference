# Custom Inference Service 

## Usage

```bash
# To check what models are currently registered in mlflow
python check_models.py

# Check what is in mlflow
python check_mlflow_artifacts.py

# To run the service
export MLFLOW_TRACKING_URI=https://ard-mlflow.slac.stanford.edu
export MODEL_NAME=lcls-cu-inj-model
export MODEL_VERSION=1
python inference_service.py
```
In another terminal, test with curl

```bash
curl http://localhost:8000/ | python -m json.tool

# Test health check
curl http://localhost:8000/health | python -m json.tool

# Get model info
curl http://localhost:8000/model/info | python -m json.tool

# Test inputs
curl http://localhost:8000/inputs | python -m json.tool

# Test outputs
curl http://localhost:8000/outputs | python -m json.tool

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "QUAD:IN20:425:BACT": -1.0,
      "SOLN:IN20:121:BACT": 0.5
    }
  }' | python -m json.tool


# To load another model
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lcls-fel-surrogate",
    "model_version": "1"
  }' | python -m json.tool
```
Can also work on gui at this link - http://127.0.0.1:8000/docs#/