import pandas as pd

# 1. Load CSV
df = pd.read_csv("./backend/models/groundwater_ap_2014_20232.csv")

# 2. Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 3. Extract year
df["year"] = df["Date"].dt.year

# 4. Keep only required columns
df = df[[
    "LATITUDE",
    "LONGITUDE",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "WL(mbgl)",
    "year"
]]

# 5. Rename columns for frontend
df = df.rename(columns={
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "DISTRICT": "district",
    "BLOCK": "block",
    "VILLAGE": "village",
    "WL(mbgl)": "wl"
})

# 6. Remove invalid rows
df = df.dropna(subset=["latitude", "longitude", "year", "wl"])

# 7. Ensure correct data types
df["year"] = df["year"].astype(int)
df["wl"] = df["wl"].astype(float)

# 8. Sort (optional, but clean)
df = df.sort_values(by=["year", "district", "block"])

# 9. Export ALL YEARS to JSON
df.to_json(
    "frontend/wells_data_years.json",
    orient="records",
    indent=2
)

# 10. Basic verification
print("JSON created successfully")
print("Total records:", len(df))
print("Years available:", sorted(df["year"].unique()))
