"""
backend/main.py
===============
FastAPI server exposing the Medical RAG chatbot API.

Endpoints:
  POST /chat          — main RAG Q&A
  GET  /health        — health check + index stats
  GET  /suggestions   — example questions
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.rag_pipeline import get_pipeline, RAGResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── App Lifecycle 
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up RAG pipeline...")
    try:
        get_pipeline()._lazy_init()
        log.info("Pipeline warm.")
    except Exception as e:
        log.error(f"Pipeline init failed: {e}")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Medical RAG Chatbot API",
    description="Citation-aware Q&A over 20K+ PubMed research articles",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, example="What are the latest treatments for pancreatic cancer?")
    top_k:    int = Field(default=6,  ge=1, le=15, description="Number of chunks to retrieve")

class CitationOut(BaseModel):
    pmid:    str
    title:   str
    authors: str
    journal: str
    year:    str
    url:     str
    mesh:    str
    score:   float

class ChatResponse(BaseModel):
    answer:          str
    citations:       list[CitationOut]
    query:           str
    num_chunks_used: int
    latency_ms:      float

class HealthResponse(BaseModel):
    status:      str
    index_name:  str
    total_vectors: int
    model:       str

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def home():
    return FileResponse(os.path.join("frontend", "index.html"))

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, summary="Ask a medical question")
async def chat(req: ChatRequest):
    """
    Submit a medical question and receive a citation-grounded answer
    synthesized from PubMed literature.
    """
    start = time.perf_counter()
    try:
        pipeline = get_pipeline()
        # Override top_k if custom value
        import backend.rag_pipeline as rp
        rp.TOP_K = req.top_k

        result: RAGResponse = await pipeline.aquery(req.question)
        latency = (time.perf_counter() - start) * 1000

        return ChatResponse(
            answer          = result.answer,
            citations       = [CitationOut(**c.to_dict()) for c in result.citations],
            query           = result.query,
            num_chunks_used = result.num_chunks_used,
            latency_ms      = round(latency, 1),
        )
    except Exception as e:
        log.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Returns service health and Pinecone index statistics."""
    try:
        from pinecone import Pinecone
        pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx   = pc.Index(os.getenv("PINECONE_INDEX_NAME", "medical-pubmed-rag"))
        stats = idx.describe_index_stats()
        return HealthResponse(
            status        = "ok",
            index_name    = os.getenv("PINECONE_INDEX_NAME", "medical-pubmed-rag"),
            total_vectors = stats.get("total_vector_count", 0),
            model         = os.getenv("LLM_MODEL", "gpt-4o"),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/suggestions", summary="Example questions")
async def suggestions():
    """Return curated example questions to help users get started."""
    return {
        "suggestions": [
            "What are the most effective immunotherapy approaches for non-small cell lung cancer?",
            "How does metformin affect cardiovascular outcomes in type 2 diabetes?",
            "What is the evidence for CRISPR gene therapy in sickle cell disease?",
            "What are the neurological complications of COVID-19?",
            "What are the current treatment guidelines for Alzheimer's disease?",
            "How effective are GLP-1 agonists for weight loss in obese patients?",
            "What is the role of gut microbiome in mental health disorders?",
            "What are the latest developments in CAR-T cell therapy?",
            "How does antibiotic resistance develop and spread?",
            "What are the risk factors for cardiovascular disease in young adults?",
        ]
    }
