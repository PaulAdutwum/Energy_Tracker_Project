import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Dorms and Appliances
dorms = ["Smith Hall", "Adams Hall", "Chu Hall", "Kaplanis Hall",
         "Gillespie Hall", "Village", "Parker Hall", "Moulton House"]
appliances = ["Phones", "Fridges", "Fans", "Cooking Wares", "Kettles",
              "Lamps", "Laptops", "Heaters", "Air Conditioners"]

# Seasons and Weather Impacts
seasons = ["Winter", "Spring", "Summer", "Fall"]
weather_impact = {"Winter": 1.3, "Spring": 1.0, "Summer": 1.5, "Fall": 1.1}

# Generate Timestamp Data (Hourly for 5 Years)
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 1, 1)
timestamps = [start_date + timedelta(hours=i) for i in range((end_date - start_date).days * 24)]

# Add holidays for simulation
holidays = [
    datetime(2023, 12, 25),  # Christmas
    datetime(2023, 11, 23),  # Thanksgiving
    datetime(2023, 7, 4),    # Independence Day
]

# Simulate Data
data = []
for timestamp in timestamps:
    dorm = random.choice(dorms)
    appliance = random.choice(appliances)
    season = seasons[(timestamp.month - 1) // 3]
    peak_factor = 1.5 if 18 <= timestamp.hour <= 22 else 1.0  # Evening peak usage
    holiday_factor = 0.7 if timestamp.date() in [holiday.date() for holiday in holidays] else 1.0  # Reduced on holidays
    base_usage = np.random.uniform(0.1, 2.0)  # Base usage in kWh
    weather_factor = weather_impact[season]
    usage = round(base_usage * peak_factor * weather_factor * holiday_factor, 2)
    
    # Add occasional spikes for anomaly detection
    if random.random() < 0.005:  # 0.5% chance of spike
        usage *= 10

    data.append([timestamp, dorm, appliance, season, usage])

# Create DataFrame
columns = ["Timestamp", "Dorm", "Appliance", "Season", "Usage (kWh)"]
df = pd.DataFrame(data, columns=columns)

# Save Full Dataset to CSV
df.to_csv("energy_data_full.csv", index=False)
print("Simulated data saved to 'energy_data_full.csv'")

# Create ML-Ready Dataset
df['Hour'] = df['Timestamp'].dt.hour
df['Month'] = df['Timestamp'].dt.month
df['DayOfWeek'] = df['Timestamp'].dt.weekday
ml_ready_df = df[["Dorm", "Appliance", "Season", "Hour", "Month", "DayOfWeek", "Usage (kWh)"]]

# Save ML-Ready Dataset
ml_ready_df.to_csv("ml_ready_energy_data.csv", index=False)
print("ML-ready data saved to 'ml_ready_energy_data.csv'")

# Visualizing Trends and Patterns
import matplotlib.pyplot as plt
import seaborn as sns

# Energy Usage by Dorms
plt.figure(figsize=(12, 6))
sns.barplot(data=df.groupby("Dorm")["Usage (kWh)"].sum().reset_index(), x="Dorm", y="Usage (kWh)", palette="viridis")
plt.title("Total Energy Usage by Dorm")
plt.xticks(rotation=45)
plt.ylabel("Total Energy Usage (kWh)")
plt.xlabel("Dorms")
plt.tight_layout()
plt.savefig("energy_usage_by_dorm.png")
print("Visualization saved as 'energy_usage_by_dorm.png'")

# Energy Usage Over Time
plt.figure(figsize=(12, 6))
df.groupby(df['Timestamp'].dt.date)["Usage (kWh)"].sum().plot(color="orange")
plt.title("Energy Usage Over Time")
plt.ylabel("Total Usage (kWh)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("energy_usage_over_time.png")
print("Visualization saved as 'energy_usage_over_time.png'")

# Machine Learning Example with Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare Features and Target
X = ml_ready_df[["Hour", "Month", "DayOfWeek"]]
y = ml_ready_df["Usage (kWh)"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Deployment with Flask
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Energy Tracker Dashboard!"

@app.route("/usage-summary", methods=["GET"])
def usage_summary():
    summary = df.groupby("Dorm")["Usage (kWh)"].sum().to_dict()
    return jsonify(summary)

if __name__ == "__main__":
    app.run(debug=True)