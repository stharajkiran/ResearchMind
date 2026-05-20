FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv
ENV UV_TORCH_BACKEND=cpu
RUN uv sync --no-dev
CMD ["sh", "-c", "uv run uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
