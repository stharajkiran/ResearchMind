import logging
from pathlib import Path

def configure_logging(log_file: Path, logger: logging.Logger) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear pre-existing handlers to avoid duplicate log lines on repeated runs.
    if logger.handlers:
        logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)