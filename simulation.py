import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, jsonify


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
end_date = datetime(2024, 1, 1)
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
df['Year'] = df['Timestamp'].dt.year
ml_ready_df = df[["Dorm", "Appliance", "Season", "Hour", "Month", "DayOfWeek", "Usage (kWh)"]]

# Save ML-Ready Dataset
ml_ready_df.to_csv("ml_ready_energy_data.csv", index=False)
print("ML-ready data saved to 'ml_ready_energy_data.csv'")

# Visualizing Trends and Patterns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import seaborn as sns

# Energy Usage by Dorms
plt.figure(figsize=(12, 6))
sns.barplot(data=df.groupby(["Dorm", "Season"])["Usage (kWh)"].mean().reset_index(),
            x="Dorm", y="Usage (kWh)", hue="Season", palette="muted")
plt.title("Average Energy Usage by Dorm and Season")
plt.ylabel("Average Usage (kWh)")
plt.xlabel("Dorms")
plt.legend(title="Season")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("energy_usage_by_dorm_and_season.png")
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
#Enegy Usage by dorm and hours
plt.figure(figsize=(12, 6))
sns.lineplot(data=df.groupby(["Hour", "Dorm"])["Usage (kWh)"].mean().reset_index(),
             x="Hour", y="Usage (kWh)", hue="Dorm", palette="tab10")
plt.title("Average Energy Usage by Hour and Dorm")
plt.ylabel("Average Usage (kWh)")
plt.xlabel("Hour of the Day")
plt.legend(title="Dorm")
plt.tight_layout()
plt.savefig("energy_usage_by_hour_and_dorm.png")

#Energy Usage by Dorm and Year

plt.figure(figsize=(12, 6))
sns.barplot(data=df.groupby(["Year", "Dorm"])["Usage (kWh)"].sum().reset_index(),
            x="Year", y="Usage (kWh)", hue="Dorm", palette="Set2")
plt.title("Total Energy Usage by Dorm and Year")
plt.ylabel("Total Usage (kWh)")
plt.xlabel("Year")
plt.legend(title="Dorm")
plt.tight_layout()
plt.savefig("energy_usage_by_year_and_dorm.png")

# Energy Usage by Dorm and Appliance
plt.figure(figsize=(12, 6))
sns.barplot(data=df.groupby(["Dorm", "Appliance"])["Usage (kWh)"].mean().reset_index(),
            x="Dorm", y="Usage (kWh)", hue="Appliance", palette="coolwarm")
plt.title("Average Energy Usage by Dorm and Appliance")
plt.ylabel("Average Usage (kWh)")
plt.xlabel("Dorm")
plt.legend(title="Appliance", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("energy_usage_by_dorm_and_appliance.png")

# 1. Highest Appliance Usage Per Dorm
plt.figure(figsize=(12, 6))
appliance_usage = df.groupby(['Dorm', 'Appliance'])['Usage (kWh)'].sum().reset_index()
sns.barplot(data=appliance_usage, x='Dorm', y='Usage (kWh)', hue='Appliance', palette='muted')
plt.title("Highest Appliance Usage by Dorm")
plt.xlabel("Dorm")
plt.ylabel("Total Energy Usage (kWh)")
plt.legend(title="Appliance", loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("highest_appliance_usage_by_dorm.png")
plt.show()

plt.figure(figsize=(12, 6))
peak_usage = df.groupby(['Dorm', 'Hour'])['Usage (kWh)'].mean().reset_index()
sns.lineplot(data=peak_usage, x='Hour', y='Usage (kWh)', hue='Dorm', palette='tab10')
plt.title("Peak Usage Times by Dorm")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Usage (kWh)")
plt.legend(title="Dorm", loc='upper right')
plt.tight_layout()
plt.savefig("peak_usage_times_by_dorm.png")
plt.show()

# 3. Highest Energy Consumption Throughout the Year for Each Dorm
plt.figure(figsize=(12, 6))
annual_usage = df.groupby(['Dorm', 'Year'])['Usage (kWh)'].sum().reset_index()
sns.barplot(data=annual_usage, x='Year', y='Usage (kWh)', hue='Dorm', palette='coolwarm')
plt.title("Annual Energy Consumption by Dorm")
plt.xlabel("Year")
plt.ylabel("Total Energy Usage (kWh)")
plt.legend(title="Dorm", loc='upper right')
plt.tight_layout()
plt.savefig("annual_energy_consumption_by_dorm.png")
plt.show()

print("Visualizations saved successfully!")




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
    return render_template("index.html")
@app.route("/visualizations")
def visualizations():
    return jsonify({
        "Energy Usage by Dorm and Season": "energy_usage_by_dorm_and_season.png",
        "Energy Usage by Hour and Dorm": "energy_usage_by_hour_and_dorm.png",
        "Energy Usage by Dorm and Year": "energy_usage_by_year_and_dorm.png",
        "Energy Usage by Dorm and Appliance": "energy_usage_by_dorm_and_appliance.png"
    })

if __name__ == "__main__":
    app.run(debug=True)