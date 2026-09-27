FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STOCKINTEL_DB=/data/stockintel.db HOME=/data
COPY pyproject.toml README.md ./
COPY stockintel ./stockintel
RUN pip install --no-cache-dir . && mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000
# STOCKINTEL_API_KEY must be set at run time; the server refuses to start without it.
CMD ["sh", "-c", "stockintel serve --host 0.0.0.0 --port ${PORT:-8000}"]
