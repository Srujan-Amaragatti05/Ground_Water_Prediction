import pandas as pd
import requests
import json
import os
import time

INPUT_FILE = "D:\\NewE\\College\\Project\\GW\\dataset\\dataset30.csv"
CACHE_FILE = "D:\\NewE\\College\\Project\\GW\\dataset\\weather_cache.json"
OUTPUT_FILE = "D:\\NewE\\College\\Project\\GW\\dataset\\fixed_real_dataset.csv"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_FILE, low_memory=False)

df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# ==============================
# GRID (IMPORTANT)
# ==============================
df['LAT_API'] = (df['LATITUDE'] / 0.5).round() * 0.5
df['LON_API'] = (df['LONGITUDE'] / 0.5).round() * 0.5

# ==============================
# LOAD CACHE
# ==============================
if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    print(f"✅ Loaded cache: {len(cache)}")
else:
    cache = {}
    print("🆕 Starting fresh cache")

# ==============================
# API FUNCTION
# ==============================
def get_weather_bulk(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime('%Y%m%d'),
        "end": end.strftime('%Y%m%d'),
        "format": "JSON"
    }

    try:
        res = requests.get(url, params=params, timeout=30)
        data = res.json()['properties']['parameter']
        return data
    except Exception as e:
        print("API Error:", e)
        return None

# ==============================
# FETCH WEATHER (IMPORTANT PART)
# ==============================
groups = df.groupby(['LAT_API', 'LON_API'])

print("📍 Locations:", len(groups))

for idx, ((lat, lon), group) in enumerate(groups):

    start = group['Date'].min()
    end = group['Date'].max()

    print(f"🌍 [{idx+1}/{len(groups)}] {lat},{lon}")

    data = get_weather_bulk(lat, lon, start, end)
    if data is None:
        continue

    for date in group['Date']:
        d = date.strftime('%Y%m%d')
        key = f"{lat}_{lon}_{d}"

        if key not in cache:
            cache[key] = {
                "temp": data['T2M'].get(d),
                "humidity": data['RH2M'].get(d),
                "rain": data['PRECTOTCORR'].get(d)
            }

    # save periodically
    if idx % 20 == 0:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print("💾 Cache saved")

    time.sleep(0.3)

# final save
with open(CACHE_FILE, "w") as f:
    json.dump(cache, f)

print("✅ API Fetch Done!")

# ==============================
# FILL DATASET
# ==============================
df['Temperature'] = None
df['Humidity'] = None
df['Rainfall'] = None

missing = 0

for i, row in df.iterrows():

    lat = round(row['LAT_API'], 1)
    lon = round(row['LON_API'], 1)
    d = row['Date'].strftime('%Y%m%d')

    key = f"{lat}_{lon}_{d}"

    if key in cache:
        df.at[i, 'Temperature'] = cache[key]['temp']
        df.at[i, 'Humidity'] = cache[key]['humidity']
        df.at[i, 'Rainfall'] = cache[key]['rain']
    else:
        missing += 1

    if i % 50000 == 0:
        print(f"📊 Filled {i} rows...")

print("❌ Missing:", missing)

# ==============================
# SAVE
# ==============================
df.to_csv(OUTPUT_FILE, index=False)

print("🎉 DONE")