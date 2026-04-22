pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-repo/insurance-fraud-detection.git'
            }
        }
        stage('Setup') {
            steps {
                sh 'python -m venv venv'
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }
        stage('Generate Data & Train') {
            steps {
                sh 'mkdir -p data models'
                sh 'python generate_data.py'
                sh 'python train_model.py'
            }
        }
        stage('Test') {
            steps {
                sh '. venv/bin/activate && python -c "import app, pickle; print(\'Models loaded OK\')"'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t fraud-detection:latest .'
            }
        }
        stage('Deploy (Local)') {
            steps {
                sh 'docker-compose up -d'
            }
        }
    }
    post {
        always {
            sh 'docker-compose down'
        }
    }
}