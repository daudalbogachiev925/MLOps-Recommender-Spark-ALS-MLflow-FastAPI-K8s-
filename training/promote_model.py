"""Переводит последнюю версию модели в стадию Production."""
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow:5000")
name = "als_recommender"

versions = client.search_model_versions(f"name='{name}'")
latest = max(versions, key=lambda v: int(v.version))
client.transition_model_version_stage(
    name=name, version=latest.version, stage="Production", archive_existing_versions=True,
)
print(f"✅ {name} v{latest.version} → Production")
