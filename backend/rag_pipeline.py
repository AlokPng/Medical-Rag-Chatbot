"""
backend/rag_pipeline.py
=======================
Core LangChain-based RAG pipeline — 100% FREE stack.
  • Embeddings : HuggingFace sentence-transformers (local, no API key)
  • LLM        : Groq API (free tier) — llama-3.3-70b-versatile
  • Vector DB  : Pinecone Serverless
"""

import os
import re
import logging
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pinecone import Pinecone

load_dotenv()
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-pubmed-rag")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL        = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
TOP_K            = int(os.getenv("RETRIEVAL_TOP_K", 6))


# ─── Citation Dataclass ───────────────────────────────────────────────────────
@dataclass
class Citation:
    pmid:    str
    title:   str
    authors: str
    journal: str
    year:    str
    url:     str
    mesh:    str = ""
    score:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "pmid":    self.pmid,
            "title":   self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year":    self.year,
            "url":     self.url,
            "mesh":    self.mesh,
            "score":   round(self.score, 4),
        }


@dataclass
class RAGResponse:
    answer:    str
    citations: list[Citation] = field(default_factory=list)
    query:     str = ""
    num_chunks_used: int = 0

    def to_dict(self) -> dict:
        return {
            "answer":          self.answer,
            "citations":       [c.to_dict() for c in self.citations],
            "query":           self.query,
            "num_chunks_used": self.num_chunks_used,
        }


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedRAG, an expert medical research assistant. You answer questions using only the provided PubMed research articles.

Instructions:
1. Base your answer STRICTLY on the retrieved context. Do not add unsupported claims.
2. Cite sources using [PMID: XXXXX] inline whenever you reference a fact.
3. If multiple articles support a point, cite all relevant ones.
4. If the context does not contain enough information to answer, say so clearly.
5. Structure longer answers with clear sections.
6. Use precise medical terminology but briefly explain technical terms.
7. End with a "Key Takeaway" summary sentence.
8. NEVER give treatment recommendations or replace professional medical advice.

Format: Answer in clear paragraphs with inline citations like [PMID: 12345678].
"""

HUMAN_TEMPLATE = """RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Provide a thorough, citation-grounded answer based on the above context."""


# ─── RAG Pipeline ────────────────────────────────────────────────────────────
class MedicalRAGPipeline:
    def __init__(self):
        self._embeddings  = None
        self._vectorstore = None
        self._llm         = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize clients only when first needed."""
        if self._initialized:
            return

        log.info("Initializing MedicalRAGPipeline (FREE stack: Groq + HuggingFace)...")

        # HuggingFace embeddings — runs locally, no API key needed
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info(f"Loaded embedding model: {EMBEDDING_MODEL}")

        # Pinecone vectorstore via LangChain
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self._vectorstore = PineconeVectorStore(
            index=pc.Index(INDEX_NAME),
            embedding=self._embeddings,
            text_key="text",
        )

        # Groq LLM — free tier, very fast inference
        self._llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,   # Low temp for factual medical content
            max_tokens=2048,
        )
        log.info(f"Loaded LLM: {LLM_MODEL} via Groq")

        self._initialized = True
        log.info("Pipeline ready (FREE stack).")

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Format retrieved docs into context string with citation markers."""
        parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            header = (
                f"[Article {i}] PMID: {m.get('pmid','?')} | "
                f"{m.get('title','')[:80]} | "
                f"{m.get('authors','')[:50]} | "
                f"{m.get('journal','')[:40]} ({m.get('year','')})"
            )
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_cited_pmids(self, answer: str) -> list[str]:
        """Extract all PMIDs referenced in the answer."""
        return re.findall(r"\[PMID:\s*(\d+)\]", answer)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def query(self, question: str) -> RAGResponse:
        """Run RAG pipeline on a question."""
        self._lazy_init()

        # Retrieve docs with scores
        docs_with_scores = self._vectorstore.similarity_search_with_score(
            question, k=TOP_K
        )
        print("\n" + "="*50)
        print("QUESTION:", question)
        print("DOCS RETRIEVED:", len(docs_with_scores))

        for i, (doc, score) in enumerate(docs_with_scores):
            print("\nDOC", i+1)
            print("SCORE:", score)
            print("CONTENT:", repr(doc.page_content[:300]))
            print("METADATA:", doc.metadata)
        print("="*50)

        if not docs_with_scores:
            return RAGResponse(
                answer="I could not find relevant medical literature for your question. Please try rephrasing.",
                query=question,
            )

        docs   = [d for d, _ in docs_with_scores]
        scores = [s for _, s in docs_with_scores]

        # Format context
        context = self._format_docs(docs)

        # Generate answer via Groq
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_TEMPLATE),
        ])
        chain  = prompt | self._llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # Build citation objects from retrieved metadata
        cited_pmids = set(self._extract_cited_pmids(answer))
        seen_pmids  = set()
        citations   = []

        for doc, score in zip(docs, scores):
            m    = doc.metadata
            pmid = m.get("pmid", "")
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            if pmid in cited_pmids or score > 0.75:
                citations.append(Citation(
                    pmid    = pmid,
                    title   = m.get("title", ""),
                    authors = m.get("authors", ""),
                    journal = m.get("journal", ""),
                    year    = m.get("year", ""),
                    url     = m.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                    mesh    = m.get("mesh_terms", ""),
                    score   = float(score),
                ))

        return RAGResponse(
            answer           = answer,
            citations        = sorted(citations, key=lambda c: -c.score),
            query            = question,
            num_chunks_used  = len(docs),
        )

    async def aquery(self, question: str) -> RAGResponse:
        """Async version — runs sync query in executor for FastAPI."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question)


# Singleton
_pipeline: Optional[MedicalRAGPipeline] = None

def get_pipeline() -> MedicalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalRAGPipeline()
    return _pipeline
