import json
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sklearn.decomposition import PCA
from auth import get_user, get_user_by_id, create_user, User

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['SESSION_TYPE'] = 'filesystem'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

# Load PCA and sample points for charts
with open('models/pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('models/cluster_sample_points.json', 'r') as f:
    cluster_sample_points = json.load(f)
with open('models/cluster_fraud_rates.json', 'r') as f:
    cluster_fraud_rates_json = json.load(f)

# Load ML models and scaler
with open('models/isolation_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)
with open('models/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('models/cluster_fraud_rates.pkl', 'rb') as f:
    cluster_fraud_rates = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_fraud(features_dict):
    features = np.array([[
        features_dict['age'],
        features_dict['vehicle_age'],
        features_dict['past_claims'],
        features_dict['num_cars'],
        features_dict['days_accident'],
        features_dict['days_claim'],
        features_dict['driver_rating']
    ]])
    features_scaled = scaler.transform(features)
    pca_coords = pca.transform(features_scaled)[0]
    pca_x, pca_y = float(pca_coords[0]), float(pca_coords[1])

    if_pred = iso_forest.predict(features_scaled)[0]
    anomaly_score = iso_forest.score_samples(features_scaled)[0]
    norm_anomaly = max(0, min(1, (-anomaly_score)/0.5))
    cluster = kmeans.predict(features_scaled)[0]
    cluster_fraud_prob = cluster_fraud_rates[cluster]
    is_fraud = (if_pred == -1) or (cluster_fraud_prob > 0.5)
    confidence = (norm_anomaly + cluster_fraud_prob) / 2.0
    return is_fraud, confidence, cluster, pca_x, pca_y

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('fraud_detection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('fraud_detection'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and user['password'] == password:
            user_obj = User(user['id'], user['username'])
            login_user(user_obj)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('fraud_detection'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('fraud_detection'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if create_user(username, password):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists', 'danger')
    return render_template('register.html')

@app.route('/fraud')
@login_required
def fraud_detection():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        age = int(request.form['age'])
        vehicle_age = int(request.form['vehicle_age'])
        past_claims = int(request.form['past_claims'])
        num_cars = int(request.form['num_cars'])
        days_accident = int(request.form['days_accident'])
        days_claim = int(request.form['days_claim'])
        driver_rating = int(request.form['driver_rating'])

        features_dict = {
            'age': age,
            'vehicle_age': vehicle_age,
            'past_claims': past_claims,
            'num_cars': num_cars,
            'days_accident': days_accident,
            'days_claim': days_claim,
            'driver_rating': driver_rating
        }
        is_fraud, confidence, cluster, pca_x, pca_y = predict_fraud(features_dict)

        result = {
            'fraud': bool(is_fraud),
            'confidence': round(confidence, 3),
            'cluster': int(cluster),
            'message': '🚨 FRAUD DETECTED 🚨' if is_fraud else '✅ CLAIM LOOKS LEGITIMATE ✅',
            'pca_x': pca_x,
            'pca_y': pca_y
        }

        # Store in session for dashboard display
        session['prediction'] = result

        # Store in history
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        history = session['prediction_history']
        history.insert(0, {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'result': result,
            'features': features_dict
        })
        session['prediction_history'] = history[:10]

        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('fraud_detection'))

@app.route('/dashboard')
@login_required
def dashboard():
    import pandas as pd
    import numpy as np
    import json

    df = pd.read_csv('data/claims_with_clusters.csv')
    total_claims = len(df)
    fraud_count = df['fraud_label'].sum()
    fraud_percent = round(fraud_count / total_claims * 100, 2)

    cluster_counts = df['cluster'].value_counts().sort_index().tolist()
    cluster_labels = [f'Cluster {i}' for i in sorted(df['cluster'].unique())]
    cluster_fraud_rates = df.groupby('cluster')['fraud_label'].mean().multiply(100).round(2).tolist()

    X_pca = np.load('models/X_pca.npy')
    scatter_data = {}
    for cluster_id in sorted(df['cluster'].unique()):
        idx = df[df['cluster'] == cluster_id].index
        points = X_pca[idx].tolist()
        if len(points) > 500:
            points = points[:500]
        scatter_data[cluster_id] = points

    cluster_colors = {0: '#3b82f6', 1: '#10b981', 2: '#f59e0b', 3: '#ef4444'}

    prediction = session.pop('prediction', None)

    return render_template('dashboard.html',
                           stats={'total_claims': total_claims, 'fraud_count': fraud_count, 'fraud_percent': fraud_percent},
                           cluster_labels=json.dumps(cluster_labels),
                           cluster_counts=json.dumps(cluster_counts),
                           cluster_fraud_rates=json.dumps(cluster_fraud_rates),
                           scatter_data=scatter_data,
                           cluster_colors=cluster_colors,
                           prediction=prediction)

@app.route('/history')
@login_required
def history():
    history_list = session.get('prediction_history', [])
    return render_template('history.html', history=history_list)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/api/cluster_data')
@login_required
def cluster_data():
    data = {
        'sample_points': {int(k): v for k, v in cluster_sample_points.items()},
        'fraud_rates': {int(k): v for k, v in cluster_fraud_rates_json.items()},
        'cluster_labels': list(cluster_fraud_rates_json.keys())
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)