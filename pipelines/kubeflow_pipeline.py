"""Kubeflow Pipelines: prepare → train → promote → deploy."""
from kfp import dsl, compiler
from kfp.dsl import component, Output, Artifact

@component(base_image="python:3.11-slim", packages_to_install=["pandas", "pyarrow"])
def prepare(out: Output[Artifact]):
    import pandas as pd, random
    random.seed(42)
    rows = [(random.randint(1, 5000), random.randint(1, 1000),
             round(random.uniform(1, 5), 1)) for _ in range(100_000)]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).drop_duplicates()
    df.to_parquet(out.path)
    out.metadata["rows"] = str(len(df))

@component(base_image="bitnami/spark:3.5")
def train(dataset: dsl.Input[Artifact]):
    # placeholder — вызывай spark-submit train_als.py
    print("Training on", dataset.path)

@component(base_image="python:3.11-slim", packages_to_install=["mlflow"])
def promote():
    from mlflow.tracking import MlflowClient
    c = MlflowClient("http://mlflow:5000")
    vs = c.search_model_versions("name='als_recommender'")
    latest = max(vs, key=lambda v: int(v.version))
    c.transition_model_version_stage("als_recommender", latest.version, "Production", True)

@dsl.pipeline(name="recommender-pipeline")
def pipeline():
    p = prepare()
    t = train(p.outputs["out"])
    promote().after(t)

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "recommender_pipeline.yaml")
    print("✅ compiled recommender_pipeline.yaml")
