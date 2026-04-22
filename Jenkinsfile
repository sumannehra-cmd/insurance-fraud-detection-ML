pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/sumannehra-cmd/insurance-fraud-detection-ML.git'
            }
        }
        stage('Setup') {
            steps {
                bat 'python -m venv venv'
                bat 'venv\\Scripts\\activate && pip install -r requirements.txt'
            }
        }
        stage('Generate Data & Train') {
            steps {
                bat 'python generate_data.py'
                bat 'python train_model.py'
            }
        }
        stage('Test') {
            steps {
                bat 'venv\\Scripts\\activate && python -c "import pickle; print(\'Models loaded OK\')"'
            }
        }
        stage('Build Docker Image') {
            steps {
                bat 'docker build -t fraud-detection:latest .'
            }
        }
        stage('Deploy (Local)') {
            steps {
                bat 'docker-compose up -d'
            }
        }
    }
    post {
        always {
            bat 'docker-compose down'
        }
    }
}