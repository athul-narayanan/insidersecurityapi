import pandas as pd
import os

# Path to the ExtractedData folder
data_path = 'ExtractedData/'

# List all pickle files in ExtractedData (e.g., '1.pickle', '2.pickle', ...)
pickle_files = [f for f in os.listdir(data_path) if f.endswith('.pickle')]

dfs = []
for pf in pickle_files:
    df = pd.read_pickle(os.path.join(data_path, pf))
    dfs.append(df)

# Concatenate all weekly data
weekly_data = pd.concat(dfs, ignore_index=True)

# Check columns (should include 'user' and 'insider')
print(weekly_data.columns)
print(weekly_data.head())

# Get unique malicious users
malicious_users = weekly_data[weekly_data['insider'] == 1]['user'].unique().tolist()
print(f"Malicious users found: {malicious_users}")