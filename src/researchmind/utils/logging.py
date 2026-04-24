import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime



@dataclass
class DownloadMetrics:
    total: int
    completed: int = 0
    failed: int = 0
    start_time: datetime = None
    
    def log_progress(self, logger: logging.Logger):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta_sec = (self.total - self.completed) / rate if rate > 0 else 0
        logger.info(
            f"Download: {self.completed}/{self.total} | "
            f"Failed: {self.failed} | "
            f"Rate: {rate:.1f} items/sec | "
            f"ETA: {eta_sec:.0f}s"
        )

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

def configure_logging_root(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


