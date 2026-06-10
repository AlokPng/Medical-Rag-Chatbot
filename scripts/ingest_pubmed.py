"""
scripts/ingest_pubmed.py
========================
Fetches PubMed articles via NCBI Entrez API, chunks them,
generates FREE HuggingFace embeddings (local), and upserts into Pinecone.

Usage:
    python scripts/ingest_pubmed.py --query "cancer immunotherapy" --max 5000
    python scripts/ingest_pubmed.py --resume   # continues from last checkpoint

 uses sentence-transformers locally.
"""

import os
import json
import time
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Generator

import requests
import xmltodict
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import tiktoken

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

#  Config 
NCBI_EMAIL          = os.getenv("NCBI_EMAIL", "user@example.com")
NCBI_API_KEY        = os.getenv("NCBI_API_KEY", "")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
INDEX_NAME          = os.getenv("PINECONE_INDEX_NAME", "medical-pubmed-rag")
PINECONE_ENV        = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM       = 384   # all-MiniLM-L6-v2 output dimension (free, local)
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", 64))
EMBED_BATCH         = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
UPSERT_BATCH        = int(os.getenv("UPSERT_BATCH_SIZE", 100))

CHECKPOINT_FILE     = Path("data/ingest_checkpoint.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
import os

print("CWD:", os.getcwd())
print("EMAIL:", repr(os.getenv("NCBI_EMAIL")))
print("API_KEY:", repr(os.getenv("NCBI_API_KEY")))

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Medical search queries for broad coverage
DEFAULT_QUERIES = [
    "cancer immunotherapy treatment",
    "cardiovascular disease risk factors",
    "diabetes mellitus management",
    "COVID-19 clinical outcomes",
    "neurodegenerative disease Alzheimer Parkinson",
    "antibiotic resistance bacteria infection",
    "mental health depression anxiety treatment",
    "vaccine efficacy clinical trial",
    "gene therapy CRISPR disease",
    "obesity metabolic syndrome",
]

# ─── Clients 
# HuggingFace embeddings 
hf_embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},   # change to "cuda" a GPU
    encode_kwargs={"normalize_embeddings": True},
)

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index() -> any:
    """Create Pinecone index if it doesn't exist, return index object."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        log.info(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,   # 384 for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        log.info("Index created and ready.")
    else:
        log.info(f"Using existing index '{INDEX_NAME}'.")
    return pc.Index(INDEX_NAME)


# ─── Checkpoint 
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"indexed_pmids": [], "total_chunks": 0}

def save_checkpoint(state: dict):
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


# ─── PubMed Fetching 
def _ncbi_params(extra: dict) -> dict:
    params = {"email": NCBI_EMAIL, "tool": "MedRAGChatbot", **extra}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def search_pubmed(query: str, max_results: int = 500, retstart: int = 0) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    resp = requests.get(
        f"{ENTREZ_BASE}/esearch.fcgi",
        params=_ncbi_params({
            "db": "pubmed", "term": query,
            "retmax": max_results, "retstart": retstart,
            "retmode": "json", "sort": "relevance",
        }),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch full abstracts for a list of PMIDs via efetch."""
    if not pmids:
        return []
    resp = requests.post(
        f"{ENTREZ_BASE}/efetch.fcgi",
        data=_ncbi_params({
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "xml",
        }),
        timeout=60,
    )
    resp.raise_for_status()
    parsed = xmltodict.parse(resp.text)
    articles_raw = parsed.get("PubmedArticleSet", {}).get("PubmedArticle", [])
    if isinstance(articles_raw, dict):  
        articles_raw = [articles_raw]

    articles = []
    for item in articles_raw:
        try:
            medline = item["MedlineCitation"]
            article = medline["Article"]
            pmid    = str(medline["PMID"]["#text"] if isinstance(medline["PMID"], dict) else medline["PMID"])
            title   = article.get("ArticleTitle", "")
            if isinstance(title, dict):
                title = title.get("#text", "")

            # Abstract text — can be structured or flat
            abstract_obj = article.get("Abstract", {})
            abstract_text_raw = abstract_obj.get("AbstractText", "") if abstract_obj else ""
            if isinstance(abstract_text_raw, list):
                abstract_text = " ".join(
                    (t.get("#text", "") if isinstance(t, dict) else str(t))
                    for t in abstract_text_raw
                )
            elif isinstance(abstract_text_raw, dict):
                abstract_text = abstract_text_raw.get("#text", "")
            else:
                abstract_text = str(abstract_text_raw)

            # Authors
            author_list = article.get("AuthorList", {}).get("Author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            authors = []
            for a in author_list[:5]:
                ln = a.get("LastName", "")
                fn = a.get("ForeName", "")
                if ln:
                    authors.append(f"{ln} {fn}".strip())

            # Journal + year
            journal_info = article.get("Journal", {})
            journal_name = journal_info.get("Title", "")
            pub_date     = journal_info.get("JournalIssue", {}).get("PubDate", {})
            year         = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4] if isinstance(pub_date.get("MedlineDate", ""), str) else "")

            # MeSH terms
            mesh_list = medline.get("MeshHeadingList", {})
            mesh_terms = []
            if mesh_list:
                headings = mesh_list.get("MeshHeading", [])
                if isinstance(headings, dict):
                    headings = [headings]
                for h in headings:
                    d = h.get("DescriptorName", {})
                    if isinstance(d, dict):
                        mesh_terms.append(d.get("#text", ""))
                    else:
                        mesh_terms.append(str(d))

            if abstract_text.strip():
                articles.append({
                    "pmid":     pmid,
                    "title":    str(title),
                    "abstract": abstract_text,
                    "authors":  authors,
                    "journal":  journal_name,
                    "year":     str(year),
                    "mesh":     mesh_terms[:10],
                    "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        except Exception as e:
            log.debug(f"Skipping article parse error: {e}")
    return articles


# ─── Chunking 
enc = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_article(article: dict) -> list[dict]:
    """Split article into overlapping chunks with rich metadata."""
    full_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
    chunks    = splitter.split_text(full_text)
    result    = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{article['pmid']}_chunk_{i}".encode()).hexdigest()
        result.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                "text": chunk,   # IMPORTANT

                "pmid": article["pmid"],
                "title": article["title"][:300],
                "authors": ", ".join(article["authors"][:3]),
                "journal": article["journal"][:100],
                "year": article["year"],
                "url": article["url"],
                "mesh_terms": ", ".join(article["mesh"][:5]),
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        })
    return result


# Embedding 
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using local HuggingFace model — completely FREE."""
    return hf_embedder.embed_documents(texts)


#  Main Ingestion Loop 
def batched(lst: list, n: int) -> Generator:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def ingest(queries: list[str], max_total: int, resume: bool = False):
    index     = get_or_create_index()
    state     = load_checkpoint() if resume else {"indexed_pmids": [], "total_chunks": 0}
    seen_pmids = set(state["indexed_pmids"])
    total_chunks = state["total_chunks"]

    per_query = max(100, max_total // len(queries))
    all_pmids: list[str] = []

    log.info(f"Searching PubMed across {len(queries)} queries, ~{per_query} each...")
    for query in queries:
        pmids = search_pubmed(query, max_results=per_query)
        new   = [p for p in pmids if p not in seen_pmids]
        all_pmids.extend(new)
        log.info(f"  '{query[:50]}': {len(new)} new PMIDs")
        time.sleep(0.4)

    all_pmids = list(dict.fromkeys(all_pmids))[:max_total]
    log.info(f"Total new PMIDs to process: {len(all_pmids)}")

    fetch_size = 100  # NCBI allows up to 500 but 100 is safer
    pbar = tqdm(total=len(all_pmids), desc="Ingesting articles", unit="article")

    for pmid_batch in batched(all_pmids, fetch_size):
        articles = fetch_abstracts(pmid_batch)
        all_chunks = []
        for article in articles:
            all_chunks.extend(chunk_article(article))

        # Embed in sub-batches using local HuggingFace model (no rate limits)
        texts = [c["text"] for c in all_chunks]
        vectors = []
        for text_batch in batched(texts, EMBED_BATCH):
            embeddings = embed_texts(text_batch)
            vectors.extend(embeddings)

        # Build Pinecone records
        records = [
            {"id": c["id"], "values": v, "metadata": c["metadata"]}
            for c, v in zip(all_chunks, vectors)
        ]

        # Upsert to Pinecone
        for upsert_batch in batched(records, UPSERT_BATCH):
            index.upsert(vectors=upsert_batch)

        # Update state
        indexed_pmids_batch = [a["pmid"] for a in articles]
        seen_pmids.update(indexed_pmids_batch)
        total_chunks += len(all_chunks)
        state = {"indexed_pmids": list(seen_pmids), "total_chunks": total_chunks}
        save_checkpoint(state)

        pbar.update(len(pmid_batch))
        time.sleep(0.5)  # Be polite to NCBI

    pbar.close()
    log.info(f" Ingestion complete! {len(seen_pmids)} articles, {total_chunks} chunks indexed.")
    index_stats = index.describe_index_stats()
    log.info(f"Pinecone index stats: {index_stats}")


# ─── CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed articles into Pinecone")
    parser.add_argument("--query",  type=str, nargs="+", help="Custom search queries")
    parser.add_argument("--max",    type=int, default=int(os.getenv("PUBMED_MAX_ARTICLES", 20000)))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    queries = args.query if args.query else DEFAULT_QUERIES
    ingest(queries=queries, max_total=args.max, resume=args.resume)
