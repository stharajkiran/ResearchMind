from contextlib import asynccontextmanager
import logging
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from researchmind.utils.logging import configure_logging
from .retriever import RetrieverService
from .models import SearchRequest, SearchResult
from researchmind.utils.find_root import find_project_root

project_root = find_project_root()
# build logger for the API module
logger = logging.getLogger(__name__)


def configure_api_logging():
    logs_dir = project_root / "logs" / "api"
    log_path = logs_dir / f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(log_path, logger)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging once at startup
    configure_api_logging()
    logger.info("API startup")

    # Load retriever
    app.state.retriever = RetrieverService(project_root)
    app.state.retriever.load()
    yield
    logger.info("API shutdown")


app = FastAPI(lifespan=lifespan)
# Set up Prometheus instrumentation for metrics collection
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.exception_handler(ResponseValidationError)
async def validation_exception_handler(request: Request, exc: ResponseValidationError):
    logger.exception("Response validation failed")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest, request: Request) -> list[SearchResult]:
    try:
        retriever = request.app.state.retriever
        return retriever.search(req.query, req.k)
    except Exception:
        logger.exception("Search failed for query: %s", req.query)
        raise
