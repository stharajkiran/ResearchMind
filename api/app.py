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
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.rrf import reciprocal_rank_fusion


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


project_root = find_project_root()
# build logger for the API module
logger = logging.getLogger("api")

client = instructor.from_anthropic(
    Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
)


def configure_api_logging():
    logs_dir = project_root / "logs" / "api"
    log_path = logs_dir / f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(log_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging once at startup
    configure_api_logging()
    logger.info("API startup")

    # Load retriever
    app.state.retriever = RetrieverService(project_root)
    app.state.retriever.load()

    # load chromadb client and collection if needed for RAG
    app.state.chroma_store = ChromaStore(
        "researchmind", encoder=app.state.retriever.encoder
    )

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
def search(req: SearchRequest, request: Request) -> list[Chunk]:
    """Search for relevant chunks given a query.

    Args:
        req (SearchRequest): The search request containing the query and number of results.
        request (Request): The FastAPI request object.

    Returns:
        list[Chunk]: A list of chunks matching the search query.
    """
    try:
        logger.info("Received search request: %s", req.query)
        retriever = request.app.state.retriever
        results = retriever.search(req.query, req.k)
        logger.info("Search completed successfully for query")
        return results
    except Exception:
        logger.exception("Search failed for query: %s", req.query)
        raise


@app.post("/rag")
def rag(req: RAGRequest, request: Request) -> RAGResponse:
    """Handle a RAG (Retrieval-Augmented Generation) request.

    Args:
        req (RAGRequest): The RAG request containing the query and other parameters.
        request (Request): The FastAPI request object.

    Returns:
        RAGResponse: The RAG response containing the generated answer and other metadata.
    """
    try:
        logger.info("Received RAG query: %s", req.query)
        # retreive from chromadb as list of chunks
        chroma_results = request.app.state.chroma_store.query_collection(
            query=req.query,
            n_results=10,
        )
        # lis of chunks from retriever search: RFF(faiss + bm25)
        bm25_faiss_results = request.app.state.retriever.search(req.query, k=10)

        logger.info("Retrieved %d chunks from Chroma and %d chunks from Retriever for RAG query.", len(chroma_results), len(bm25_faiss_results))
         # RRF merge — need ranked lists of chunk_ids
        chroma_ids = [c.chunk_id for c in chroma_results]
        bm25_ids = [c.chunk_id for c in bm25_faiss_results]
        merged_ids = reciprocal_rank_fusion(chroma_ids, bm25_ids)[:5]

        logger.info("Chroma IDs: %s", chroma_ids[:3])
        logger.info("Retriever IDs: %s", bm25_ids[:3])
        logger.info("Merged IDs: %s", merged_ids)

        # build lookup from both result sets
        all_chunks = {c.chunk_id: c for c in chroma_results + bm25_faiss_results}
        # filter the merged chunks
        final_chunks = [all_chunks[cid] for cid in merged_ids if cid in all_chunks]

        logger.info("Retrieved %d chunks from Chroma for RAG query.", len(final_chunks))
        # build prompt with query and retrieved chunks and call Anthropic API with prompt, return  RaGResponse

        SYSTEM_PROMPT, content = build_prompt(req.query, final_chunks)
        # return  RaGResponse
        logger.info(
            "Constructed system prompt and content for RAG query. Sending to Anthropic API..."
        )
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
