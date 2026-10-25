import os
import time
import requests
import pandas as pd

# Perioada de analiza
DATA_START = "2024-01-01"
DATA_END = "2026-10-26"  # exclusiv

API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("No OPENAQ_API_KEY environment variable found.")

URL = "https://api.openaq.org/v3/sensors/{}/days"

SENSORS = {
    "B-1": [8623290, 8613102],
    "B-2": [7774357, 8622321],
    "B-3": [7775432, 8613414],
    "B-4": [7774145, 8613353],
    "B-5": [7774981, 8613434],
    "B-6": [7774877, 8613466],
    "B-7": [7774212, 8613564],
}

HEADERS = {"X-API-Key": API_KEY}
rows = []

for station, sensor_list in SENSORS.items():
    for sensor_id in sensor_list:
        print(f"Descarc {station}, senzor {sensor_id}...")
        page = 1

        while True:
            params = {"date_from": DATA_START, "date_to": DATA_END, "limit": 1000, "page": page}
            response = requests.get(URL.format(sensor_id), headers=HEADERS, params=params, timeout=60)

            if response.status_code != 200:
                print(f"  Eroare HTTP {response.status_code}: {response.text[:300]}")
                break

            results = response.json().get("results", [])
            if not results:
                break

            for item in results:
                period = item.get("period") or {}
                start = period.get("datetimeFrom") or {}
                local = start.get("local")
                utc = start.get("utc")
                date = local or utc

                rows.append({
                    "statie": station,
                    "sensor_id": sensor_id,
                    "date": date,
                    "no2_ug_m3": item.get("value"),
                    "unitate": (item.get("parameter") or {}).get("units"),
                    "has_flags": (item.get("flagInfo") or {}).get("hasFlags"),
                })

            if len(results) < 1000:
                break

            page += 1
            time.sleep(0.2)

        time.sleep(0.2)

df = pd.DataFrame(rows)

if df.empty:
    print("Nu au fost returnate masuratori.")
else:
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].dt.tz_convert("Europe/Bucharest").dt.strftime("%Y-%m-%d")
    df["no2_ug_m3"] = pd.to_numeric(df["no2_ug_m3"], errors="coerce")
    df = df.dropna(subset=["no2_ug_m3"])
    df = df[df["unitate"] == "µg/m³"]
    df.to_csv("openaq_no2_bucuresti_2024_2026.csv", index=False)