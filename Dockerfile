# ==============================================================================
# 🎓 STUDENT PERFORMANCE PREDICTION - PRODUCTION DOCKERFILE
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sed -i '/-e \./d' requirements.txt && pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "application.py"]