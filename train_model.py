
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "hourly_power.csv"
MODEL_PATH = BASE_DIR / "powerwise_model.joblib"
PLOT_PATH = BASE_DIR / "actual_vs_predicted.png"

print("Loading hourly data...")

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()

# Create time-based features
df["day_of_week"] = df.index.day_of_week
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# Previous readings
for lag in [1, 2, 3, 24, 48, 168]:
    df[f"lag_{lag}"] = df["Global_active_power"].shift(lag)

# The target is power usage 24 hours ahead
df["target"] = df["Global_active_power"].shift(-24)

df = df.dropna()

feature_cols = [
    "hour", "day_of_week", "month",
    "lag_1", "lag_2", "lag_3",
    "lag_24", "lag_48", "lag_168"
]

X = df[feature_cols]
y = df["target"]

# Chronological split: earliest 80% for training, latest 20% for testing
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Baseline: use the same hour's reading from the previous day
baseline_predictions = X_test["lag_24"].to_numpy()

print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

def report_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    return mae, rmse

baseline_mae, baseline_rmse = report_metrics(
    "Previous-day baseline", y_test, baseline_predictions
)

model_mae, model_rmse = report_metrics(
    "Random Forest", y_test, predictions
)

# Save the trained model
joblib.dump(
    {"model": model, "features": feature_cols},
    MODEL_PATH
)

# Plot a sample of test predictions
plt.figure(figsize=(12, 5))
plt.plot(
    y_test.index[:168],
    y_test.iloc[:168],
    label="Actual",
    linewidth=2
)
plt.plot(
    y_test.index[:168],
    predictions[:168],
    label="Random Forest prediction",
    linewidth=2
)
plt.title("PowerWise: Actual vs Predicted Power Usage")
plt.xlabel("Time")
plt.ylabel("Average active power (kW)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()

print("\nSaved model to:", MODEL_PATH)
print("Saved prediction plot to:", PLOT_PATH)
print("\nTraining and evaluation complete!")
