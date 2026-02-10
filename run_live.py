import time
from dust_model import DustNowcastSystem

system = DustNowcastSystem()

new_rows_added = 0

while True:
    base, roads = system.nowcast_latest(retrain=False)

    # count new rows written to CSV
    new_rows_added += 1   # 1 row per hour fetch (adjust if different)

    # ✅ retrain only after enough NEW data
    if new_rows_added >= 24:   # ~1 day of hourly data
        system.train_ml_until_latest()
        new_rows_added = 0     # reset counter

    time.sleep(600)  # 10 minutes
