import pandas as pd
import requests
import time

df = pd.read_csv("D:\\NewE\\College\\Project\\GW\\dataset\\groundwater_ap_2014_20232.csv")
df['Date'] = pd.to_datetime(df['Date'])

def get_weather(lat, lon, date):
    start = date.strftime('%Y%m%d')
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": start,
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        values = data['properties']['parameter']

        temp = list(values['T2M'].values())[0]
        rain = list(values['PRECTOTCORR'].values())[0]
        humidity = list(values['RH2M'].values())[0]

        print(f"Fetched weather for {lat}, {lon} on {date.date()}: Temp={temp}, Humidity={humidity}, Rainfall={rain}")  
        return temp, humidity, rain

    except Exception as e:
        return None, None, None


temps, hums, rains = [], [], []

for i, row in df.iterrows():
    lat = row['LATITUDE']
    lon = row['LONGITUDE']

    if pd.notnull(lat) and pd.notnull(lon):
        t, h, r = get_weather(lat, lon, row['Date'])
    else:
        t, h, r = None, None, None

    temps.append(t)
    hums.append(h)
    rains.append(r)

    if i % 50 == 0:
        print(f"Processed {i} rows...")
        time.sleep(1)

df['Temperature'] = temps
df['Humidity'] = hums
df['Rainfall'] = rains

df.to_csv("fixed_real_dataset.csv", index=False)

print("✅ Done!")