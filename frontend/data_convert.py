import pandas as pd

# Load dataset
df = pd.read_csv("./backend/models/groundwater_ap_2014_20232.csv")

# Find latest year
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
latest_year = df["year"].max()
print("Latest year:", latest_year)

# Filter latest year
df_latest = df[df["year"] == latest_year]

# Keep only required columns
df_latest = df_latest[[
    "LATITUDE", "LONGITUDE",
    "DISTRICT", "BLOCK", "VILLAGE",
    "WL(mbgl)"
]]

# Rename columns to match frontend
df_latest = df_latest.rename(columns={
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "DISTRICT": "district",
    "BLOCK": "block",
    "VILLAGE": "village",
    "WL(mbgl)": "wl"
})

# Save to JSON
df_latest.to_json(
    "frontend/wells_data2.json",
    orient="records",
    indent=2
)

print("JSON created successfully")
