FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r backend/requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]