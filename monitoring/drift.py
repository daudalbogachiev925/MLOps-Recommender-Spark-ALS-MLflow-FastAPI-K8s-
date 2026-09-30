"""Проверка дрифта признаков через Evidently и алерт в вебхук."""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Пример: два датасета признаков (reference = обучение, current = последние N дней)
ref = pd.read_parquet("/opt/data/ratings_ref.parquet")
cur = pd.read_parquet("/opt/data/ratings_cur.parquet")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("/tmp/drift_report.html")

result = report.as_dict()
drift = result["metrics"][0]["result"]["dataset_drift"]
print(f"🚨 dataset_drift={drift}")
# Здесь можно дёрнуть alertmanager/webhook
