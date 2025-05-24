import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

# Create cache directory if it doesn't exist
os.makedirs("f1_cache_new", exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache("f1_cache_new")# Set up logging to display progress and errors

# Load FastF1 2024 Monaco GP race session

session_2024 = fastf1.get_session(2024, "Monaco", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Qualifying Data Chinese GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Charles Leclerc", "Oscar Piastri", "Lewis Hamilton", "Max Verstappen", 
               "Isack Hadjar", "Fernando Alonso", "Esteban Ocon", "Liam Lawson", "Alexander Albon", 
                "Carlos Sainz Jr.", "Yuki Tsunoda", "Nico Hülkenberg", "George Russell", "Andrea Kimi Antonelli",  
               "Gabriel Bortoleto", "Oliver Bearman", "Pierre Gasly", "Lance Stroll", "Franco Colapinto"],
    "QualifyingTime (s)": [69.954, 70.063, 70.129, 70.382, 70.669,
                           70.923, 70.924, 70.942, 71.129, 71.213,
                           71.362, 71.415, 71.596, 71.507, 71.880,
                           71.902, 71.979, 71.994, 72.563, 72.597]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Franco Colapinto": "COL", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Create y as a DataFrame with driver code and average lap time
avg_lap_times = laps_2024.groupby("Driver")[["LapTime (s)"]].mean().reset_index()

# Merge y into merged_data using DriverCode ↔ Driver (from avg_lap_times)
final_data = merged_data.merge(avg_lap_times, left_on="DriverCode", right_on="Driver", how="inner")

# Define features (X) and target (y) using only rows where we have both
X = final_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
y = final_data["LapTime (s)"]

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
final_data["PredictedRaceTime (s)"] = predicted_race_times

final_results = final_data.sort_values("PredictedRaceTime (s)")[["Driver_x", "PredictedRaceTime (s)"]]
final_results.rename(columns={"Driver_x": "Driver"}, inplace=True)

print("\n🏁 Predicted 2025 Monaco GP Winner with Available Data 🏁\n")
print(final_results)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")