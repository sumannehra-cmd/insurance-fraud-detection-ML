FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python generate_data.py
RUN python train_model.py
RUN python init_db.py
EXPOSE 5000
CMD ["python", "app.py"]