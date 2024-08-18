# client demo

from mlflow.tracking import MlflowClient
import mlflow
# Initialize the MLflow Client
client = MlflowClient()

# Replace with the run_id of the run where the model was logged
run_id = "1031570268e5468e95e5607b520d728e"

# Replace with the path to the logged model within the run
model_path = "file:///D:/MLOps/mlflow_demo/mlruns/124059855347149989/1031570268e5468e95e5607b520d728e/artifacts/rf"

# Construct the model URI
model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the model registry
model_name = "diabetes_rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)