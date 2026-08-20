FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

RUN mkdir -p data/raw data/processed/models

CMD ["python", "-u", "src/pipeline.py"]