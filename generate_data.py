import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000
fraud_ratio = 0.05
n_fraud = int(n_samples * fraud_ratio)
n_normal = n_samples - n_fraud

claim_amount_norm = np.random.uniform(100, 10000, n_normal)
incident_hour_norm = np.random.randint(0, 24, n_normal)
incident_severity_norm = np.random.randint(1, 4, n_normal)
num_vehicles_norm = np.random.randint(1, 3, n_normal)
age_policyholder_norm = np.random.randint(18, 70, n_normal)
prior_claims_norm = np.random.randint(0, 3, n_normal)

claim_amount_fraud = np.random.uniform(20000, 50000, n_fraud)
incident_hour_fraud = np.random.randint(0, 24, n_fraud)
incident_severity_fraud = np.random.randint(3, 6, n_fraud)
num_vehicles_fraud = np.random.randint(2, 5, n_fraud)
age_policyholder_fraud = np.random.randint(18, 70, n_fraud)
prior_claims_fraud = np.random.randint(3, 6, n_fraud)

claim_amount = np.concatenate([claim_amount_norm, claim_amount_fraud])
incident_hour = np.concatenate([incident_hour_norm, incident_hour_fraud])
incident_severity = np.concatenate([incident_severity_norm, incident_severity_fraud])
num_vehicles = np.concatenate([num_vehicles_norm, num_vehicles_fraud])
age_policyholder = np.concatenate([age_policyholder_norm, age_policyholder_fraud])
prior_claims = np.concatenate([prior_claims_norm, prior_claims_fraud])
fraud_label = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

indices = np.arange(n_samples)
np.random.shuffle(indices)
df = pd.DataFrame({
    'claim_amount': claim_amount[indices],
    'incident_hour': incident_hour[indices],
    'incident_severity': incident_severity[indices],
    'num_vehicles': num_vehicles[indices],
    'age_policyholder': age_policyholder[indices],
    'prior_claims': prior_claims[indices],
    'fraud_label': fraud_label[indices]
})

df.to_csv('data/insurance_data.csv', index=False)
print("✅ Dataset saved to data/insurance_data.csv")
print(df.head())