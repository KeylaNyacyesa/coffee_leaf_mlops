FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WORKERS=1

WORKDIR /app

# Install the Python packages first, so this step is cached and only re-runs when
# requirements-docker.txt changes. No TensorFlow is needed (see src/serving.py).
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --retries 10 --default-timeout 120 -r requirements-docker.txt

COPY src ./src
COPY api ./api
COPY ui ./ui
COPY models ./models
COPY data ./data

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
