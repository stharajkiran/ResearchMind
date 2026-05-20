FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv
ENV UV_TORCH_BACKEND=cpu
RUN uv sync --no-dev
CMD ["celery", "-A", "worker.tasks", "worker", "--pool=eventlet", "-c", "4", "--loglevel=info"]
