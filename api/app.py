from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator


from researchmind.embedding.models import MPNetEncoder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.retrieval.retriever import RetrieverService
from researchmind.session.cache import QueryCache
from researchmind.session.memory import SessionMemory
from researchmind.session.redis_backend import RedisCache
from .models import SearchRequest, RAGRequest
from researchmind.ingestion.models import RAGResponse
from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging_root
from researchmind.utils.build_prompt import build_prompt
from researchmind.ingestion.models import Chunk
from researchmind.utils.llm_client import ResearchMindLLM

from researchmind.agent.graph import build_graph
from researchmind.agent.tracing import configure_tracing
from researchmind.graph.citation_graph import load_graph
from researchmind.guardrails.pipeline import ValidatorPipeline
from researchmind.feedback.store import FeedbackStore
from api.models import FeedbackRequest

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


project_root = find_project_root()
logger = logging.getLogger("api")

phase_config = load_phase_config(project_root)
_logs_dir = project_root / "logs" / "api"
_log_path = _logs_dir / f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def configure_api_logging():
    _logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging_root(_log_path)


client = ResearchMindLLM(
    tiers={tier: (tc.model, tc.provider) for tier, tc in phase_config.llm_tiers.items()}
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging once at startup
    configure_api_logging()
    logger.info("API startup")
    logger.info("Loading retriever with phase=%s", phase_config.phase)
    logger.info("Artifact dir: %s", phase_config.artifact_dir)
    logger.info("Chunks path: %s", phase_config.chunks_path)

    # Wire up retrieval backends
    encoder = MPNetEncoder()
    dense = FaissIndexBuilder(
        dimension=encoder.dim,
        artifact_dir=phase_config.artifact_dir,
        index_type=phase_config.index_type,
    )
    dense.load()
    sparse = BM25IndexBuilder(artifact_dir=phase_config.artifact_dir)
    sparse.load()
    filtered = ChromaStore(collection_name="researchmind", encoder=encoder)

    app.state.retriever = RetrieverService(
        dense=dense,
        sparse=sparse,
        filtered=filtered,
        encoder=encoder,
        chunks_path=phase_config.chunks_path,
    )
    app.state.paper_metadata = app.state.retriever.lookup_paper_metadata

    # Postgres connection and feedback store setup
    app.state.store = FeedbackStore()
    app.state.store.create_tables()

    # shared Redis backend — both cache and session memory use the same connection
    redis_backend = RedisCache()
    app.state.session_memory = SessionMemory(backend=redis_backend)
    app.state.query_cache = QueryCache(backend=redis_backend)

    configure_tracing()
    citation_graph = load_graph(project_root / "artifacts" / "citation_graph.pkl")
    pipeline = ValidatorPipeline(
        app.state.retriever.corpus_paper_ids, app.state.retriever.encoder
    )
    app.state.agent = build_graph(
        retriever=app.state.retriever,
        llm=client,  # the ResearchMindLLM already created
        citation_graph=citation_graph,
        pipeline=pipeline,
        store=app.state.store,
        session_memory=app.state.session_memory,
        query_cache=app.state.query_cache,
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
        results = retriever.search(
            req.query,
            req.k,
            mode=req.retrieval_mode,
            recency_decay_rate=req.recency_decay,
        )
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
        # lis of chunks from retriever search: RFF(faiss + bm25)
        bm25_faiss_chunks = request.app.state.retriever.search(
            req.query,
            k=5,
            mode=req.retrieval_mode,
            recency_decay_rate=req.recency_decay,
        )
        # extract chunk texts for context in RAGAS
        contexts = [c.text for c in bm25_faiss_chunks]

        logger.info(
            "Retrieved %d chunks from  Retriever for RAG query.", len(bm25_faiss_chunks)
        )
        # build prompt with query and retrieved chunks and call Anthropic API with prompt, return  RaGResponse

        SYSTEM_PROMPT, content = build_prompt(req.query, bm25_faiss_chunks)
        # return  RaGResponse
        logger.info(
            "Constructed system prompt and content for RAG query. Sending to Anthropic API..."
        )
        response = client.complete_structured(
            user_prompt=content,
            response_model=RAGResponse,
            tier="best",  # maps to the LLM used for this tier
            system_prompt=SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.0,  # some creativity allowed for synthesis
        )
        # the chunk texts that were actually retrieved by your system at query time
        response.contexts = contexts

        logger.info("Received response from Anthropic API for RAG query.")
        return response
    except Exception:
        logger.exception("RAG failed for query: %s", req.query)
        raise


@app.post("/agent")
def agent(req: RAGRequest, request: Request) -> dict:
    result = request.app.state.agent.invoke(
        {
            "query": req.query,
            "intent": "",
            "retrieved_chunks": [],
            "compared_chunks": None,
            "tool_call_history": [],
            "session_id": "",
            "final_answer": None,
            "feedback_id": None,
        }
    )
    return {"answer": result["final_answer"], "feedback_id": result.get("feedback_id")}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, request: Request):
    """Endpoint to receive user feedback on the generated answers.

    Args:
        req (FeedbackRequest): The feedback request containing the feedback ID and rating.
        request (Request): The FastAPI request object.

    Returns:
        dict: A response indicating success or failure of feedback submission.
    """
    try:
        logger.info("Received feedback submission: %s", req)
        request.app.state.store.update_rating(req.feedback_id, req.rating)
        logger.info(
            "Feedback submitted successfully for feedback ID: %d", req.feedback_id
        )
        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception:
        logger.exception(
            "Failed to submit feedback for feedback ID: %d", req.feedback_id
        )
        return {"status": "error", "message": "Failed to submit feedback."}


@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str, request: Request) -> dict:
    """Endpoint to check the status of a paper ingestion task.

    Args:
        task_id (str): The ID of the ingestion task to check.
        request (Request): The FastAPI request object.

    Returns:
        dict: A response containing the status of the ingestion task.
    """
    try:
        logger.info("Checking status for ingestion task ID: %s", task_id)
        from celery.result import AsyncResult
        from worker.tasks import celery_app

        # Check the status of the Celery task using the task ID
        result = AsyncResult(task_id, app=celery_app)
        logger.info("Ingestion task ID: %s has status: %s", task_id, result.status)
        return {"task_id": task_id, "status": result.status, "result": result.result}
    except Exception:
        logger.exception("Failed to check status for ingestion task ID: %s", task_id)
        return {"status": "error", "message": "Failed to check ingestion status."}


@app.get("/paper/{paper_id}")
def get_paper(paper_id: str, request: Request):
    meta = request.app.state.paper_metadata.get(paper_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"paper_id": paper_id, **meta}
