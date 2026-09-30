"""Обучение ALS с MLflow tracking + регистрация модели."""
import mlflow, mlflow.spark, sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT = "recommender-als"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

spark = (
    SparkSession.builder
    .appName("ALS-Trainer")
    .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.13.0")
    .getOrCreate()
)

ratings = spark.read.parquet("/opt/data/ratings.parquet").cache()
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="user_id", itemCol="item_id", ratingCol="rating",
    coldStartStrategy="drop", nonnegative=True,
)

grid = (
    ParamGridBuilder()
    .addGrid(als.rank, [10, 50])
    .addGrid(als.regParam, [0.01, 0.1])
    .addGrid(als.maxIter, [10])
    .build()
)

rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
mae_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

cv = CrossValidator(
    estimator=als, estimatorParamMaps=grid,
    evaluator=rmse_eval, numFolds=3, parallelism=3,
)

with mlflow.start_run(run_name="als_cv") as run:
    model = cv.fit(train)
    preds = model.transform(test)
    rmse = rmse_eval.evaluate(preds)
    mae = mae_eval.evaluate(preds)

    best = model.bestModel
    mlflow.log_params({
        "rank": best.rank,
        "regParam": best._java_obj.parent().getRegParam(),
        "maxIter": best._java_obj.parent().getMaxIter(),
    })
    mlflow.log_metrics({"rmse": rmse, "mae": mae})

    mlflow.spark.log_model(
        best, "model",
        registered_model_name="als_recommender",
    )
    print(f"✅ run_id={run.info.run_id} rmse={rmse:.4f} mae={mae:.4f}")

spark.stop()
