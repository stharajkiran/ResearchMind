from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator


from researchmind.retrieval.retriever import RetrieverService
from .models import SearchRequest, RAGRequest
from researchmind.ingestion.models import RAGResponse
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging_root
from researchmind.utils.build_prompt import build_prompt
from researchmind.ingestion.models import Chunk
from researchmind.utils.llm_client import ResearchMindLLM

from researchmind.agent.graph import build_graph
from researchmind.agent.tracing import configure_tracing
from researchmind.graph.citation_graph import load_graph

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def configure_api_logging():
    logs_dir = Config.logs_dir
    log_path = Config.log_path
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging_root(log_path)


project_root = find_project_root()
# build logger for the API module
logger = logging.getLogger("api")


class Config:
    "Configuration for API, including logging and any other global setup."

    phase = os.environ.get("INDEX_PHASE", "phase2")
    artifact_dir = project_root / "artifacts" / "indexes" / phase
    if phase == "semantic":
        chunks_path = (
            project_root / "data" / "processed" / "cleaned_semantic_chunks.jsonl"
        )
    else:
        chunks_path = project_root / "data" / "processed" / "cleaned_chunks.jsonl"

    # model_name = "claude-sonnet-4-6"
    model_name = "qwen3.6:27b"
    logs_dir = project_root / "logs" / "api"
    log_path = logs_dir / f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


# client = instructor.from_anthropic(
#     Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# )
client = ResearchMindLLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging once at startup
    configure_api_logging()
    logger.info("API startup")
    logger.info("Loading retriever with index phase: %s", Config.phase)
    logger.info("Using artifact dir: %s", Config.artifact_dir)
    logger.info("Using chunks path: %s", Config.chunks_path)

    # Load retriever
    app.state.retriever = RetrieverService(
        Config.artifact_dir,
        collection_name="researchmind",
        chunks_path=Config.chunks_path,
    )
    app.state.retriever.load(Config.chunks_path)

    configure_tracing()
    citation_graph = load_graph(project_root / "artifacts" / "citation_graph.pkl")
    app.state.agent = build_graph(
        retriever=app.state.retriever,
        llm=client,  # the ResearchMindLLM already created
        citation_graph=citation_graph,
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
        }
    )
    return {"answer": result["final_answer"]}
