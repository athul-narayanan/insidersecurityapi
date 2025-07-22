import pandas as pd
import os
from django.conf import settings
import joblib
import numpy as np
import torch

def group_sort(df):
    df = sort_by_date(df)
    df = handle_na(df)
    df = derive_features(df)
    df =  df.groupby(['from', 'user', 'date']).agg({
        'hour': 'mean',
        'weekday': 'first',
        'num_to': 'mean',
        'num_cc': 'mean',
        'num_bcc': 'mean',
        'num_words': 'mean',
        'has_attachment': 'sum',
        'O': 'mean',
        'C': 'mean',
        'E': 'mean',
        'A': 'mean',
        'N': 'mean'
    }).reset_index()

    return df

def pre_process(df):
    df = normalize_features(df)
    df, device = generate_sequence(df)
    return df, device
    

def handle_na(df):
    df.fillna('',inplace=True)
    return df

def sort_by_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    return df

def derive_features(df):
    # count the number of direct recipients
    # A high number may indicate mass leakage of data

    df['num_to'] = df['to'].str.split(',').apply(len)

    # count the number of cc recipients
    # A high number may indicate mass leakage of data

    df['num_cc'] = df['cc'].str.split(',').apply(len)

    # count the number of bcc recipients
    # it can be used to detect suspicious behaviour

    df['num_bcc'] = df['bcc'].str.split(',').apply(len)

    # calculate the number of words in the email
    # email with malicious intend may have higher word count

    df['num_words'] = df['content'].str.split(' ').apply(len)



    # convert attachement count into binary feature
    # attachement can be used as a clear indicator of insider threat
    # keeping attachment count as it is increased the noise

    df['has_attachment'] = df['attachments'].apply(lambda x: 1 if x > 0 else 0)

    # extract the hour (0-23) and day of the mail (0=monday and 6=Sunday)
    # Insider attacks usually happend during night and off peak day ( weekend)

    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.dayofweek

    return df

def normalize_features(df):
    scaler_path = os.path.join(settings.BASE_DIR, 'utils', 'static',  'scaler.pkl')
    scaler1_path = os.path.join(settings.BASE_DIR, 'utils', 'static',  'scaler1.pkl')

    # Load the scalers
    scaler = joblib.load(scaler_path)
    scaler1 = joblib.load(scaler1_path)

    print(df)

    scale_cols = ['hour', 'num_to', 'num_cc', 'num_bcc', 'num_words']
    psychometric_cols = ['O', 'C', 'E', 'A', 'N']
    binary_cols = ['has_attachment']

    # Normalize the data using same scalar
    scaled_part = scaler.transform(df[scale_cols])
    scaled_psych = scaler1.transform(df[psychometric_cols])
    unscaled_part = df[binary_cols].values

    # Merge the data data after normalization
    full_features = np.hstack((scaled_part, unscaled_part, scaled_psych))

    return full_features

def generate_sequence(df):
    # Use GPU if available else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequences.append(df[i:i+sequence_length])

    print("Final sequence for model evaluation is", sequences)
    X_test = np.array(sequences)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    return X_test_tensor, device

def findunique(malicious_json, top_festure):
    map = {}
    filtered_data = []
    filtered_features = []
    for mal_row, topfeature in zip(malicious_json, top_festure):
        key = mal_row["from"] + mal_row["date"].strftime("%Y-%m-%d %H:%M:%S")
        if key in map:
            continue

        filtered_data.append(mal_row)
        filtered_features.append(topfeature)
        map[key] = 1
    
    return filtered_data, filtered_features

