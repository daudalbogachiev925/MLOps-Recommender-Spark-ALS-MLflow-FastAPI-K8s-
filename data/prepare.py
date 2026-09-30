"""Генерирует синтетический датасет рейтингов для ALS."""
import random, os
import pandas as pd

random.seed(42)
N_USERS = 5000
N_ITEMS = 1000
N_RATINGS = 500_000

rows = []
for _ in range(N_RATINGS):
    u = random.randint(1, N_USERS)
    i = random.randint(1, N_ITEMS)
    # немного "скрытой структуры": любимые жанры
    base = (u * 31 + i * 17) % 5
    rating = min(5.0, max(1.0, base + random.gauss(0, 0.7)))
    rows.append((u, i, round(rating, 1)))

df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).drop_duplicates(["user_id", "item_id"])
os.makedirs("/opt/data", exist_ok=True)
df.to_parquet("/opt/data/ratings.parquet", index=False)
print(f"✅ saved {len(df)} ratings → /opt/data/ratings.parquet")
