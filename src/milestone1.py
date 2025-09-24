# milestone1.py
# Milestone 1: Data Preprocessing & EDA (Delhi 5+ Years Dataset)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Step 1: Load Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv("data/air_quality.csv")

print("\n🔎 Columns found:", df.columns.tolist())
print("First rows:\n", df.head())

# Convert 'date' column into datetime format
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

# Select pollutants (already standardized)
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
df = df[['datetime'] + pollutants]

# === Step 2: Data Preprocessing ===
print("\n🛠 Cleaning dataset...")
df = df.interpolate(method='linear')   # Fill missing values
df = df.drop_duplicates()              # Remove duplicates
df.set_index('datetime', inplace=True)

# === Step 3: Resample Data (Daily Average) ===
print("\n📊 Resampling data to daily averages...")
daily_df = df.resample('1D').mean()

# === Step 4: EDA ===
os.makedirs("outputs", exist_ok=True)

# 4.1 Time Series Trend
plt.figure(figsize=(12,6))
plt.plot(daily_df.index, daily_df['PM2.5'], label="PM2.5")
plt.plot(daily_df.index, daily_df['PM10'], label="PM10")
plt.legend()
plt.title("Daily Pollutant Trends (Delhi, 5+ years)")
plt.savefig("outputs/daily_trends.png")
plt.close()

# 4.2 Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(daily_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between Pollutants")
plt.savefig("outputs/correlation.png")
plt.close()

# 4.3 Distribution of PM2.5
daily_df['PM2.5'].hist(bins=50)
plt.title("PM2.5 Distribution (Delhi, 5+ years)")
plt.xlabel("µg/m³")
plt.ylabel("Frequency")
plt.savefig("outputs/pm25_distribution.png")
plt.close()

# 4.4 Summary Statistics
print("\n📊 Statistical Summary:")
print(daily_df.describe())

# === Step 5: Feature Engineering ===
print("\n🧩 Creating features...")
daily_df['PM2.5_lag1'] = daily_df['PM2.5'].shift(1)
daily_df['PM2.5_roll7'] = daily_df['PM2.5'].rolling(7).mean()
daily_df['day'] = daily_df.index.day
daily_df['month'] = daily_df.index.month
daily_df['year'] = daily_df.index.year

# Save processed dataset
daily_df.to_csv("data/processed_air_quality.csv")
print("\n✅ Processed dataset saved: data/processed_air_quality.csv")

print("\n🎉 Milestone 1 Completed! Check 'outputs/' for plots.")
