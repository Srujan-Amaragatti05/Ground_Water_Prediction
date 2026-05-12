# ============================================
# Groundwater Level EDA (2014–2023)
# ============================================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# ============================================
# 2. Load dataset safely (handle encoding)
# ============================================
df = pd.read_csv(r"D:\NewE\College\Project\GW\backend\models\groundwater_ap_2014_20232.csv",
    encoding="latin1",        # handles special characters
    sep=",",                  # force comma separator
    engine="python",          # python engine handles irregular rows
    on_bad_lines="skip"       # skip corrupted lines safely
)


# ============================================
# 3. Basic dataset overview
# ============================================
print("Dataset Shape (Rows, Columns):")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# ============================================
# 4. Check missing values
# ============================================
print("\nMissing Values Count:")
print(df.isna().sum())

# ============================================
# 5. Convert Date column to datetime
# ============================================
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# ============================================
# 6. Extract Year and Month
# ============================================
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# ============================================
# 7. Distribution of Groundwater Level
# ============================================
plt.figure()
sns.histplot(df["WL(mbgl)"], bins=30, kde=True) # type: ignore
plt.title("Distribution of Groundwater Level (WL in mbgl)")
plt.xlabel("Water Level (mbgl)")
plt.ylabel("Frequency")
# This plot shows how groundwater levels are spread across observations
plt.show()

# ============================================
# 8. Year-wise Average Groundwater Level Trend
# ============================================
yearly_avg = df.groupby("Year")["WL(mbgl)"].mean().reset_index()

plt.figure()
sns.lineplot(data=yearly_avg, x="Year", y="WL(mbgl)", marker="o")
plt.title("Year-wise Average Groundwater Level Trend")
plt.xlabel("Year")
plt.ylabel("Average Water Level (mbgl)")
# This plot shows how groundwater levels change over years
plt.show()

# ============================================
# 9. District-wise Average Groundwater Level
# ============================================
district_avg = (
    df.groupby("DISTRICT")["WL(mbgl)"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=district_avg, x="WL(mbgl)", y="DISTRICT")
plt.title("District-wise Average Groundwater Level")
plt.xlabel("Average Water Level (mbgl)")
plt.ylabel("District")
# This plot compares average groundwater depth across districts
plt.show()

# ============================================
# 10. Outlier Detection using Boxplot
# ============================================
plt.figure()
sns.boxplot(x=df["WL(mbgl)"])
plt.title("Outlier Detection in Groundwater Level (WL)")
plt.xlabel("Water Level (mbgl)")
# Points beyond whiskers indicate unusually high or low groundwater levels
plt.show()

# ============================================
# 11. Statistical Summary for WL
# ============================================
print("\nStatistical Summary of Groundwater Level:")
print(df["WL(mbgl)"].describe())

# ============================================
# End of EDA
# ============================================
