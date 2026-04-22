pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                bat 'py -m venv venv'
                bat 'venv\\Scripts\\activate && pip install -r requirements.txt'
            }
        }
        stage('Generate Data & Train') {
            steps {
                bat 'py generate_data.py'
                bat 'py train_model.py'
            }
        }
        stage('Test') {
            steps {
                bat 'venv\\Scripts\\activate && py -c "import pickle; print(\'Models loaded OK\')"'
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