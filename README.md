# ResearchMind — Enterprise R&D Intelligence Platform

A production-grade research assistant that retrieves, reasons over, and synthesises academic literature. Built as an open-source tool for retrieval and reasoning problem to help during litereature review and research.

---

## Why I built this

R&D teams spend roughly half their working week on literature search — finding papers, checking what cites what, figuring out what problems are still unsolved. Most existing tools return keyword matches and stop there. The expensive enterprise ones add some AI on top but lock everything behind sales calls.

This project is my attempt to build the full stack properly: multi-source retrieval, citation graph traversal, hybrid search, and LLM synthesis with actual grounding validation.

---

## Current Status — Phase 1 Complete

Phase 1 delivers a fully benchmarked hybrid search engine serving real arXiv papers over a production API.

**What's built:**
- arXiv ingestion pipeline with rate limiting and JSONL storage
- Three embedding models benchmarked — BGE selected on throughput + recall data
- Three FAISS index types benchmarked — HNSW32 selected on latency + recall data
- Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion
- FastAPI `/search` endpoint with Prometheus metrics and structured logging
- DVC pipeline — `dvc repro` reproduces the full corpus and indexes from scratch
- Load tested at 10/50/100 concurrent users

---

## Benchmarks

All benchmarks logged to MLflow. Decisions made from data, not defaults.

### Embedding model selection

Evaluated on 60 synthetic queries (30 semantic + 30 technical) over a 5,000-paper corpus. GPU-accelerated (CUDA 12.8).

| Model | Recall@10 | Semantic Recall | Technical Recall | Throughput (docs/sec) | P95 Latency | Selected |
|---|---|---|---|---|---|---|
| SPECTER2 base + adapters | 0.92 | 0.83 | 1.00 | 498 | 17.5ms | |
| bge-small-en-v1.5 | 0.95 | 0.90 | 1.00 | 1,566 | 10ms | |
| **all-mpnet-base-v2** | **0.97** | **0.93** | **1.00** | **542** | **12ms** | ✓ |

All three models hit 100% technical recall. MPNet leads on semantic recall (0.93) — the primary failure mode in research search. BGE runs at 3× the throughput but trails by 3 points on semantic recall. SPECTER2's citation-proximity training objective doesn't generalise to topic-based retrieval.

### FAISS index selection

| Index | Recall@10 | P50 Latency | P95 Latency | Build Time | Selected |
|---|---|---|---|---|---|
| Flat (brute force) | 0.97 | 4.5ms | 5.5ms | 0.026s | |
| IVF100 (inverted file) | 0.58 | <1ms | <1ms | 0.082s | |
| **HNSW32 (graph-based)** | **0.97** | **<1ms** | **0.5ms** | **0.075s** | ✓ |

IVF100 recall collapses to 0.58 despite a 5,000-paper corpus — at 50 vectors per cluster it sits at the minimum training threshold. HNSW32 matches Flat recall at sub-millisecond latency.

### Retrieval benchmark — hybrid vs single-retriever

60-query evaluation set, split by query type.

| Retriever | Semantic Recall | Technical Recall |
|---|---|---|
| FAISS only (HNSW32) | 93.3% | 100% |
| BM25 only | 73.3% | 100% |
| **RRF hybrid** | **93.3%** | **100%** |

All three hit 100% technical recall — BM25 handles exact term matching. RRF matches FAISS on semantic recall on this corpus; BM25's lower semantic recall (73.3%) reflects keyword mismatch on paraphrased queries. RRF is retained as the serving strategy — it adds no regression and improves robustness on exact-match queries.

### Load test — 100 concurrent users

| Percentile | Latency |
|---|---|
| p50 | 14ms |
| p95 | 54ms |
| p99 | 2,100ms* |

*p99 spike is encoder cold-start on first inference, not sustained degradation.

---

## Architecture

```
arXiv API ── Ingestion Pipeline ── papers.jsonl (DVC tracked)
                                        │
                              ┌─────────┴─────────┐
                         BGE Encoder          bm25s
                              │                   │
                        FAISS HNSW32         BM25 Index
                              └─────────┬─────────┘
                                   RRF Fusion
                                        │
                                 FastAPI /search
                                 (Prometheus metrics)
```

---

## Tech Stack

**Retrieval:** FAISS (HNSW32), bm25s, RRF fusion  
**Embeddings:** bge-small-en-v1.5 (benchmarked against SPECTER2, MPNet)  
**API:** FastAPI, Prometheus, uvicorn  
**Pipeline:** DVC, MLflow  
**Load testing:** Locust  
**Data:** arXiv API

---

## How to Run

```bash
git clone https://github.com/stharajkiran/ResearchMind.git
cd ResearchMind
uv venv && uv sync
cp .env.example .env

# Reproduce full pipeline (fetch papers + build indexes)
uv run dvc repro

# Start the API
uv run uvicorn api.app:app --reload
# POST http://localhost:8000/search {"query": "...", "k": 10}
# Metrics: http://localhost:8000/metrics
# Docs: http://localhost:8000/docs
```

---

## How this generalises to other domains

The core problem — professionals spending hours navigating large document collections to find answers they can act on — is not unique to research.

**Financial documents:** same retrieval and synthesis layer over filings and reports. Main changes: data sources, a finance-adapted embedding model, and a chunker that understands financial document structure.

**Legal documents:** case law has a natural citation structure that maps directly onto the citation graph component. "What later cases relied on this ruling?" is structurally identical to "what papers cited this one?"

The ingestion, retrieval, and serving layers were kept separate intentionally. Swapping the data source shouldn't require rewriting the agent or the API.

---

## Commercial context

This targets the same problem as:
- **Cypris** — 500M+ data points, enterprise R&D intelligence, $50k+/yr ARR
- **PatSnap Eureka** — GPT-powered answers grounded in patents and publications
- **Elicit** — systematic literature review for researchers

---

## Where this is going

The hybrid search engine is the foundation. The full system adds a LangGraph agent on top — seven tools including citation graph traversal (NetworkX), HyDE retrieval, and research gap detection. Every tool call is traced in LangSmith. Answers pass through guardrails-ai validators before reaching the user, so hallucinated citations never surface. Redis caches repeated queries. The whole thing ships as a Dockerised stack with a Streamlit demo..
