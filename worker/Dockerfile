FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["celery", "-A", "worker.tasks", "worker", "--pool=eventlet", "-c", "4", "--loglevel=info"]

