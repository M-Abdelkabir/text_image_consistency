FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/app.py .
# Note: Ces fichiers doivent être présents ou montés via volume si non entraînés
# COPY api/model.pkl .
# COPY api/vectorizer.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
