import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import os

# Load data
df = pd.read_csv('data/insurance_data.csv')
X = df[['claim_amount', 'incident_hour', 'incident_severity', 
        'num_vehicles', 'age_policyholder', 'prior_claims']]

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

# Compute fraud rate per cluster (using original fraud_label)
cluster_fraud_rate = df.groupby('cluster')['fraud_label'].mean().to_dict()

# Save models
os.makedirs('models', exist_ok=True)
with open('models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
with open('models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Also save cluster fraud rates for prediction
with open('models/cluster_fraud_rates.pkl', 'wb') as f:
    pickle.dump(cluster_fraud_rate, f)

print("Models trained and saved in /models")
print(f"Cluster fraud rates: {cluster_fraud_rate}")