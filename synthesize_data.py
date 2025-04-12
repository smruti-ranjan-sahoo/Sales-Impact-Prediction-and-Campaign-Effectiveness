import pandas as pd
import numpy as np
from datetime import timedelta

# Load your original dataset
df = pd.read_csv("WA_Marketing-Campaign_with_Date.csv")

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Possible values for randomization (from your dataset)
promotion_types = df['Promotion'].unique()
market_sizes = df['MarketSize'].unique()

# Number of weeks to simulate (4 months ≈ 16 weeks)
weeks_to_generate = 16

# Store all synthesized records
synthesized_data = []

for i in range(weeks_to_generate):
    temp = df.copy()
    
    # Shift date and week number
    temp['Date'] = temp['Date'] + timedelta(weeks=i)
    temp['week'] = temp['week'] + i
    
    # Add variation to Sales
    temp['SalesInThousands'] *= np.random.uniform(0.53, .99, size=len(temp))
    temp['SalesInThousands'] = temp['SalesInThousands'].round(2)
    
    # Randomly alter Promotion and MarketSize with slight variability
    temp['Promotion'] = np.random.choice(promotion_types, size=len(temp))
    temp['MarketSize'] = np.random.choice(market_sizes, size=len(temp))

    # Append to synthesized data
    synthesized_data.append(temp)

# Combine all weekly data
final_df = pd.concat(synthesized_data, ignore_index=True)

# Save to CSV
final_df.to_csv("Synthesized_4Month_Marketing.csv", index=False)

print("✅ Synthesized dataset created as 'Synthesized_4Month_Marketing.csv'")
