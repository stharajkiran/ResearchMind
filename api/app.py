from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os
from anthropic import Anthropic
import instructor

from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator


from .retriever import RetrieverService
from .models import SearchRequest, SearchResult, RAGRequest, RAGResponse
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.utils.build_prompt import build_prompt


from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


project_root = find_project_root()
# build logger for the API module
logger = logging.getLogger(__name__)

client = instructor.from_anthropic(
    Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
)


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

    # load chromadb client and collection if needed for RAG
    app.state.chroma_store = ChromaStore("researchmind", encoder=app.state.retriever.encoder)

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


@app.post("/rag")
def rag(req: RAGRequest, request: Request) -> RAGResponse:
    try:
        logger.info("Received RAG query: %s", req.query)
        # retreive from chromadb as list of chunks
        results = request.app.state.chroma_store.query_collection(
            query=req.query,
            n_results=5,
        )
        logger.info("Retrieved %d chunks from Chroma for RAG query.", len(results))
        # build prompt with query and retrieved chunks and call Anthropic API with prompt, return  RaGResponse

        SYSTEM_PROMPT, content = build_prompt(req.query, results)
        # return  RaGResponse
        logger.info("Constructed system prompt and content for RAG query. Sending to Anthropic API...")
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            system=SYSTEM_PROMPT,
            response_model=RAGResponse,
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0,  # some creativity allowed for synthesis
        )
        logger.info("Received response from Anthropic API for RAG query.")
        return response
    except Exception:
        logger.exception("RAG failed for query: %s", req.query)
        raise
