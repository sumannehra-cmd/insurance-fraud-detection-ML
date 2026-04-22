from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models
with open('models/isolation_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)
with open('models/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('models/cluster_fraud_rates.pkl', 'rb') as f:
    cluster_fraud_rates = pickle.load(f)

def predict_fraud(features):
    # features: numpy array (1,6)
    # Isolation Forest output: 1 normal, -1 anomaly
    if_pred = iso_forest.predict(features)[0]
    anomaly_score = iso_forest.score_samples(features)[0]  # more negative = more anomalous
    # Normalize anomaly score to [0,1] where 1 is highly anomalous
    # Typical range for score_samples is around -0.5 to 0.5; we clip and rescale
    norm_anomaly = max(0, min(1, ( -anomaly_score ) / 0.5 ))  # empirical

    # KMeans
    cluster = kmeans.predict(features)[0]
    cluster_fraud_prob = cluster_fraud_rates[cluster]

    # Ensemble: fraud if IF says anomaly OR cluster fraud prob > 0.5
    is_fraud = (if_pred == -1) or (cluster_fraud_prob > 0.5)
    confidence = (norm_anomaly + cluster_fraud_prob) / 2.0
    return is_fraud, confidence, cluster

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        claim_amount = float(request.form['claim_amount'])
        incident_hour = int(request.form['incident_hour'])
        incident_severity = int(request.form['incident_severity'])
        num_vehicles = int(request.form['num_vehicles'])
        age_policyholder = int(request.form['age_policyholder'])
        prior_claims = int(request.form['prior_claims'])

        features = np.array([[claim_amount, incident_hour, incident_severity,
                              num_vehicles, age_policyholder, prior_claims]])
        is_fraud, confidence, cluster = predict_fraud(features)

        result = {
            'fraud': bool(is_fraud),
            'confidence': round(confidence, 3),
            'cluster': int(cluster),
            'message': 'FRAUD DETECTED' if is_fraud else 'CLAIM LOOKS LEGITIMATE'
        }
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)