# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) system built on PubMed research articles using LangChain, Pinecone, HuggingFace embeddings, and Groq LLMs.

The application retrieves relevant medical literature from PubMed-derived vector embeddings and generates citation-aware answers grounded in scientific research.

---

# Features

* PubMed article ingestion
* Automatic text chunking
* HuggingFace local embeddings
* Pinecone vector search
* Citation-aware responses
* FastAPI backend
* Groq Llama 3.3 70B integration
* Semantic medical question answering
* Metadata-based citation cards
* Retrieval-Augmented Generation (RAG)

---

## Architecture

```text
Question
      │
      ▼
Embedding Generation
      │
      ▼
Pinecone Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Context Formatting
      │
      ▼
Groq Llama 3.3
      │
      ▼
Answer Generation
      │
      ▼
PMID Extraction
      │
      ▼
Citation Mapping
      │
      ▼
Final Response
```

# Ingestion Pipeline

```text
PubMed API
      │
      ▼
Article Collection
      │
      ▼
Chunking
(512 size / 64 overlap)
      │
      ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
      │
      ▼
Pinecone Serverless
      │
      ▼
Semantic Retrieval
```

---

# Tech Stack

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| Backend         | FastAPI                                |
| RAG Framework   | LangChain                              |
| LLM             | Groq (Llama 3.3 70B Versatile)         |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | Pinecone                               |
| Data Source     | PubMed                                 |
| Parsing         | Biopython, xmltodict                   |
| Chunking        | RecursiveCharacterTextSplitter         |
| Frontend        | HTML, CSS, JavaScript                  |
| Retry Logic     | Tenacity                               |

---

# Project Structure

```text
MEDICALBOT/

├── backend/
│   ├── rag_pipeline.py
│   └── __pycache__/
│
├── frontend/
│   └── index.html
│
├── scripts/
│   ├── ingest_pubmed.py
│   └── data/
│
├── data/
│
├── main.py
├── .env
├── requirements.txt
├── README.md
│
└── __pycache__/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd MEDICALBOT
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
# Groq
GROQ_API_KEY=

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=medical-pubmed-rag
PINECONE_ENVIRONMENT=us-east-1

# PubMed
NCBI_EMAIL=
NCBI_API_KEY=

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
RETRIEVAL_TOP_K=6

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
```

---

# Running Data Ingestion

## Ingest 200 Articles

```bash
python scripts/ingest_pubmed.py --max 200
```

## Ingest 1000 Articles

```bash
python scripts/ingest_pubmed.py --max 1000
```

## Ingest 20,000 Articles

```bash
python scripts/ingest_pubmed.py --max 20000
```

## Resume Interrupted Ingestion

```bash
python scripts/ingest_pubmed.py --resume
```

---

# Running the Backend

Start FastAPI:

```bash
uvicorn main:app --reload
```

Server:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

# Frontend

Open:

```text
frontend/index.html
```

Or serve locally:

```bash
python -m http.server 3000 --directory frontend
```

Frontend URL:

```text
http://localhost:3000
```

---

# API Endpoints

## POST /chat

Request:

```json
{
  "question": "What is cancer immunotherapy?"
}
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {
      "pmid": "31311655",
      "title": "Integrative Approaches to Cancer Immunotherapy",
      "authors": "Szeto Gregory L, Finley Stacey D",
      "journal": "Trends in Cancer",
      "year": "2019",
      "url": "https://pubmed.ncbi.nlm.nih.gov/31311655/",
      "score": 0.89
    }
  ],
  "num_chunks_used": 6
}
```

---

## GET /health

Returns:

```json
{
  "status": "healthy",
  "index": "medical-pubmed-rag"
}
```

---

## GET /suggestions

Returns sample medical research questions.

---

# Retrieval Workflow

```text
Question
    │
    ▼
Embedding Generation
    │
    ▼
Pinecone Similarity Search
    │
    ▼
Top-K Chunks
    │
    ▼
Prompt Construction
    │
    ▼
Groq Llama 3.3
    │
    ▼
Answer Generation
    │
    ▼
Citation Extraction
```

---

# Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Redis Response Caching
* LangSmith Observability
* Streaming Responses
* Query Expansion
* Re-ranking Models
* Multi-Query Retrieval
* PostgreSQL Metadata Store
* User Authentication
* Rate Limiting
* Conversation Memory

---

# Performance Notes

* Embedding Model: all-MiniLM-L6-v2
* Embedding Dimension: 384
* Vector Database: Pinecone Serverless
* Retrieval Strategy: Similarity Search
* Top K Documents: 6
* Chunk Size: 512
* Chunk Overlap: 64

---

# Disclaimer

This project is intended for educational and research purposes only.

The generated responses should not be considered medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.
