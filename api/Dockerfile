FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && \
    uv sync --no-dev && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]