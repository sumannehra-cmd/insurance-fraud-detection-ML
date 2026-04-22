from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pickle
import numpy as np
from auth import get_user, get_user_by_id, create_user, User

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # important for session

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # agar user direct fraud page pe jaye toh login page bhej de

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

# Load ML models (pehle ki tarah)
with open('models/isolation_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)
with open('models/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('models/cluster_fraud_rates.pkl', 'rb') as f:
    cluster_fraud_rates = pickle.load(f)

def predict_fraud(features):
    if_pred = iso_forest.predict(features)[0]
    anomaly_score = iso_forest.score_samples(features)[0]
    norm_anomaly = max(0, min(1, (-anomaly_score)/0.5))
    cluster = kmeans.predict(features)[0]
    cluster_fraud_prob = cluster_fraud_rates[cluster]
    is_fraud = (if_pred == -1) or (cluster_fraud_prob > 0.5)
    confidence = (norm_anomaly + cluster_fraud_prob) / 2.0
    return is_fraud, confidence, cluster

# Routes
@app.route('/')
def home():
    # Agar user login hai toh fraud detection page dikhao, otherwise login page
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
        if user and user['password'] == password:  # plain text password (simple for demo)
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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)