
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "household_power_consumption.txt"
)

OUTPUT_PATH = (
    Path(__file__).parent
    / "data"
    / "hourly_power.csv"
)

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Combining date and time...")

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df = df.dropna(subset=["datetime"])
df = df.set_index("datetime").sort_index()

# Convert power readings to numeric values
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

# Count missing power readings before handling them
missing_before = df["Global_active_power"].isna().sum()
print("Missing power readings:", missing_before)

# Fill short gaps by interpolation, then remove any remaining gaps
df["Global_active_power"] = (
    df["Global_active_power"]
    .interpolate(method="time", limit=5, limit_area="inside")
)

df = df.dropna(subset=["Global_active_power"])

print("Creating hourly averages...")

hourly = df["Global_active_power"].resample("h").mean()
hourly = hourly.dropna()

hourly = hourly.to_frame(name="Global_active_power")

hourly.to_csv(OUTPUT_PATH)

print("\nPreprocessing complete!")
print("Hourly rows:", len(hourly))
print("Date range:", hourly.index.min(), "to", hourly.index.max())
print("Remaining missing values:", hourly.isna().sum().sum())
print("Saved to:", OUTPUT_PATH)
print("\nFirst 5 hourly readings:")
print(hourly.head())
