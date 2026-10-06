FROM python:3.12-slim
WORKDIR /app
# app state, caches and downloaded weights live on the /data volume; the image holds only code
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/data \
    ALFA_DATA=/data ALFA_ARTIFACTS=/data/artifacts ALFA_FRONTEND=/app/frontend STOCKINTEL_DB=/data/stockintel.db
COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/backend ./backend/backend
COPY frontend ./frontend
RUN pip install --no-cache-dir "./backend[cloud]" && mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000
# STOCKINTEL_API_KEY must be set at run time; the server refuses to start without it.
CMD ["sh", "-c", "stockintel serve --host 0.0.0.0 --port ${PORT:-8000}"]
