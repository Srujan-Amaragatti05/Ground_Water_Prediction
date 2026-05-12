# -----------------------------------------
# Groundwater Spatial Visualization (Static)
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("groundwater_ap_2014_20232.csv")

# 2. Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 3. Filter data for a single year (2023)
df_2023 = df[df["Date"].dt.year == 2023]

# 4. Create the scatter plot
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    df_2023["LONGITUDE"],
    df_2023["LATITUDE"],
    c=df_2023["WL(mbgl)"],
    cmap="RdBu_r",      # Blue (shallow) → Red (deep)
    s=40,
    edgecolor="k",
    alpha=0.8
)

# 5. Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Water Level (mbgl)", fontsize=11)

# 6. Add title and labels
plt.title("Groundwater Levels in Andhra Pradesh (2023)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 7. Improve layout
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

# 8. Show plot
plt.show()
