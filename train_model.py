import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import re
import json

print("📊 Loading real dataset...")
df = pd.read_csv('data/fraud_oracle.csv')

# Target column
df['fraud_label'] = df['FraudFound_P']
print(f"✅ Dataset loaded: {len(df)} claims, {df['fraud_label'].sum()} fraudulent")

# ----- Conversion functions for text values -----
def convert_age_of_vehicle(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'new' in val_str:
        return 0
    match = re.search(r'(\d+)', val_str)
    if match:
        return int(match.group(1))
    return 0

def convert_past_claims(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'none' in val_str:
        return 0
    if 'more than 4' in val_str:
        return 5
    match = re.search(r'(\d+)', val_str)
    if match:
        return int(match.group(1))
    return 0

def convert_number_of_cars(val):
    if pd.isna(val):
        return 1
    val_str = str(val).lower().strip()
    if '1 vehicle' in val_str:
        return 1
    if '2 vehicles' in val_str:
        return 2
    if '3 to 4' in val_str:
        return 3
    if 'more than 4' in val_str:
        return 5
    match = re.search(r'(\d+)', val_str)
    if match:
        return int(match.group(1))
    return 1

def convert_days_policy(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'more than 30' in val_str:
        return 31
    match = re.search(r'(\d+)', val_str)
    if match:
        return int(match.group(1))
    return 0

# Apply conversions
df['AgeOfVehicle'] = df['AgeOfVehicle'].apply(convert_age_of_vehicle)
df['PastNumberOfClaims'] = df['PastNumberOfClaims'].apply(convert_past_claims)
df['NumberOfCars'] = df['NumberOfCars'].apply(convert_number_of_cars)
df['Days_Policy_Accident'] = df['Days_Policy_Accident'].apply(convert_days_policy)
df['Days_Policy_Claim'] = df['Days_Policy_Claim'].apply(convert_days_policy)

# Age column is already numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# DriverRating
if 'DriverRating' in df.columns:
    df['DriverRating'] = pd.to_numeric(df['DriverRating'], errors='coerce')
else:
    df['DriverRating'] = 0

# Select features
feature_columns = [
    'Age', 'AgeOfVehicle', 'PastNumberOfClaims',
    'NumberOfCars', 'Days_Policy_Accident', 'Days_Policy_Claim', 'DriverRating'
]

X = df[feature_columns]

# Fill missing values with median
X = X.fillna(X.median())

print(f"✅ Using features: {feature_columns}")
print("First few rows after conversion:")
print(X.head())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print("🤖 Training Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_scaled)

# Train KMeans
print("🤖 Training KMeans...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Cluster fraud rates
cluster_fraud_rate = df.groupby('cluster')['fraud_label'].mean().to_dict()
print(f"📊 Cluster fraud rates: {cluster_fraud_rate}")

# Save models and scaler
os.makedirs('models', exist_ok=True)
with open('models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
with open('models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('models/cluster_fraud_rates.pkl', 'wb') as f:
    pickle.dump(cluster_fraud_rate, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- Dashboard data (PCA and sample points) ---
print("📊 Preparing dashboard data...")

# Save full dataset with clusters for later use
df.to_csv('data/claims_with_clusters.csv', index=False)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
np.save('models/X_pca.npy', X_pca)
with open('models/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Sample points (max 200 per cluster) for scatter plot performance
sample_points = {}
for c in sorted(df['cluster'].unique()):
    cluster_indices = df[df['cluster'] == c].index
    sample_count = min(200, len(cluster_indices))
    sample_indices = np.random.choice(cluster_indices, sample_count, replace=False)
    sample_points[int(c)] = X_pca[sample_indices].tolist()

with open('models/cluster_sample_points.json', 'w') as f:
    json.dump(sample_points, f)

# Save cluster fraud rates as JSON for frontend
with open('models/cluster_fraud_rates.json', 'w') as f:
    json.dump(cluster_fraud_rate, f)

print("✅ Dashboard data saved (PCA, sample points, fraud rates).")
print("✅ All models and data saved in /models folder!")