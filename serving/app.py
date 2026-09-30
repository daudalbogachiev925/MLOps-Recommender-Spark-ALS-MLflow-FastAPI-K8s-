"""FastAPI-сервис рекомендаций с кэшем в Redis."""
import os, json, logging
import pandas as pd
import mlflow.pyfunc
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
MODEL_URI = os.getenv("MODEL_URI", "models:/als_recommender/Production")

mlflow.set_tracking_uri(MLFLOW_URI)
cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
app = FastAPI(title="Recommender API", version="1.0.0")

_model = None

def get_model():
    global _model
    if _model is None:
        log.info("Loading model %s ...", MODEL_URI)
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

@app.on_event("startup")
def warmup():
    try:
        get_model()
        log.info("Model warmed up")
    except Exception as e:
        log.warning("Warmup failed: %s", e)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    key = f"rec:{req.user_id}:{req.top_k}"
    cached = cache.get(key)
    if cached:
        return {"source": "cache", "items": json.loads(cached)}

    try:
        model = get_model()
        # pyfunc ALS-модель принимает DataFrame с user_id
        input_df = pd.DataFrame({"user_id": [req.user_id]})
        raw = model.predict(input_df)
        items = raw.head(req.top_k).to_dict(orient="records") if hasattr(raw, "head") else list(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache.setex(key, 3600, json.dumps(items))
    return {"source": "model", "items": items}
