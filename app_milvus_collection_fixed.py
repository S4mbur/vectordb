# -*- coding: utf-8 -*-

import json
import logging
import os
import re
import threading
import time
import uuid
from array import array
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import oracledb
import psycopg2
import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from psycopg2.extras import execute_values
from pypdf import PdfReader
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()


# ============================================================
# CONFIG HELPERS
# ============================================================


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def safe_sql_identifier(value: str, name: str) -> str:
    """Allow schema-qualified SQL identifiers while preventing SQL injection."""
    if not value or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$#]*(\.[A-Za-z_][A-Za-z0-9_$#]*)?", value):
        raise RuntimeError(f"Invalid SQL identifier for {name}: {value!r}")
    return value


# ============================================================
# CONFIG
# ============================================================

# PostgreSQL / pgvector
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = env_int("PG_PORT", 5432)
PG_DBNAME = os.getenv("PG_DBNAME", "vector_rag")
PG_USER = os.getenv("PG_USER", "rag_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "CHANGE_ME")
PG_TABLE = safe_sql_identifier(os.getenv("PG_TABLE", "ai_docs"), "PG_TABLE")
PG_CANDIDATE_K = env_int("PG_CANDIDATE_K", 30)
PG_IVFFLAT_PROBES = env_int("PG_IVFFLAT_PROBES", 10)

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ai_docs_milvus")
MILVUS_INSERT_BATCH_SIZE = env_int("MILVUS_INSERT_BATCH_SIZE", 64)
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "FLAT").upper()
MILVUS_HNSW_M = env_int("MILVUS_HNSW_M", 16)
MILVUS_HNSW_EF_CONSTRUCTION = env_int("MILVUS_HNSW_EF_CONSTRUCTION", 64)
MILVUS_HNSW_EF = env_int("MILVUS_HNSW_EF", 100)
MILVUS_IVF_NLIST = env_int("MILVUS_IVF_NLIST", 128)
MILVUS_IVF_NPROBE = env_int("MILVUS_IVF_NPROBE", 16)
MILVUS_OPERATION_TIMEOUT = env_int("MILVUS_OPERATION_TIMEOUT", 120)
MILVUS_INDEX_TIMEOUT = env_int("MILVUS_INDEX_TIMEOUT", 300)
MILVUS_FLUSH_AFTER_UPLOAD = env_bool("MILVUS_FLUSH_AFTER_UPLOAD", False)
MILVUS_FLUSH_TIMEOUT = env_int("MILVUS_FLUSH_TIMEOUT", 120)
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong").strip()
MILVUS_DETAILED_PROGRESS = env_bool("MILVUS_DETAILED_PROGRESS", False)
MILVUS_DIAGNOSTICS_ENABLED = env_bool("MILVUS_DIAGNOSTICS_ENABLED", False)

# Oracle AI Database / Vector Search
ORACLE_USER = os.getenv("ORACLE_USER", "AI_RAG")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "CHANGE_ME")
ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
ORACLE_TABLE = safe_sql_identifier(os.getenv("ORACLE_TABLE", "AI_DOCS_ORACLE"), "ORACLE_TABLE")
ORACLE_TARGET_ACCURACY = env_int("ORACLE_TARGET_ACCURACY", 80)
ORACLE_USE_APPROX_SEARCH = env_bool("ORACLE_USE_APPROX_SEARCH", True)

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = env_int("QDRANT_PORT", 6333)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_docs_qdrant")
QDRANT_INSERT_BATCH_SIZE = env_int("QDRANT_INSERT_BATCH_SIZE", 64)
QDRANT_WAIT = env_bool("QDRANT_WAIT", True)
QDRANT_TIMEOUT = env_int("QDRANT_TIMEOUT", 120)

# Elasticsearch
ES_URL = os.getenv("ES_URL", os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
ES_USERNAME = os.getenv("ES_USERNAME", "").strip() or None
ES_PASSWORD = os.getenv("ES_PASSWORD", "").strip() or None
ES_API_KEY = os.getenv("ES_API_KEY", "").strip() or None
ES_CA_CERTS = os.getenv("ES_CA_CERTS", "").strip() or None
ES_DEFAULT_VERIFY_CERTS = not ES_URL.lower().startswith("http://")
ES_VERIFY_CERTS = env_bool("ES_VERIFY_CERTS", ES_DEFAULT_VERIFY_CERTS)
ES_INDEX = os.getenv("ES_INDEX", "ai_docs_elasticsearch")
ES_INSERT_BATCH_SIZE = env_int("ES_INSERT_BATCH_SIZE", 64)
ES_NUM_CANDIDATES = env_int("ES_NUM_CANDIDATES", 100)
ES_REQUEST_TIMEOUT = env_int("ES_REQUEST_TIMEOUT", 120)

# Embedding API
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "").strip()
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "").strip()
REMOTE_MODEL_NAME = os.getenv("REMOTE_MODEL_NAME", "").strip()
EMBEDDING_REQUEST_TIMEOUT = env_int("EMBEDDING_REQUEST_TIMEOUT", 120)

# Chat API
REMOTE_CHAT_URL = os.getenv("REMOTE_CHAT_URL", "").strip()
REMOTE_CHAT_API_KEY = os.getenv("REMOTE_CHAT_API_KEY", "").strip() or REMOTE_API_KEY
REMOTE_CHAT_MODEL = os.getenv("REMOTE_CHAT_MODEL", "").strip()
CHAT_REQUEST_TIMEOUT = env_int("CHAT_REQUEST_TIMEOUT", 180)

# Dedicated rerank model/API
# Modes: none | api | llm
RERANK_MODE = os.getenv("RERANK_MODE", "api" if os.getenv("RERANK_API_URL") else "none").strip().lower()
RERANK_API_URL = os.getenv("RERANK_API_URL", os.getenv("REMOTE_RERANK_URL", "")).strip()
RERANK_API_KEY = os.getenv("RERANK_API_KEY", os.getenv("REMOTE_RERANK_API_KEY", REMOTE_API_KEY)).strip()
RERANK_MODEL = os.getenv("RERANK_MODEL", os.getenv("REMOTE_RERANK_MODEL", "")).strip()
# cohere/jina/generic: query + documents; tei: query + texts
RERANK_API_STYLE = os.getenv("RERANK_API_STYLE", "cohere").strip().lower()
RERANK_TIMEOUT = env_int("RERANK_TIMEOUT", 90)
RERANK_MAX_CANDIDATES = env_int("RERANK_MAX_CANDIDATES", 12)
RERANK_CHUNK_MAX_CHARS = env_int("RERANK_CHUNK_MAX_CHARS", 1200)
RERANK_FALLBACK = os.getenv("RERANK_FALLBACK", "vector").strip().lower()

# RAG / retrieval
EMBEDDING_DIM = env_int("EMBEDDING_DIM", 4096)
HALFVEC_DIM = env_int("HALFVEC_DIM", 2048)
TOP_K = env_int("TOP_K", 5)
RETRIEVAL_CANDIDATE_K = env_int("RETRIEVAL_CANDIDATE_K", 20)
SIMILARITY_THRESHOLD = env_float("SIMILARITY_THRESHOLD", 0.45)
MAX_CONTEXT_LENGTH = env_int("MAX_CONTEXT_LENGTH", 12000)
ENABLE_QUERY_REWRITE = env_bool("ENABLE_QUERY_REWRITE", True)

# Chunking
CHUNK_MAX_WORDS = env_int("CHUNK_MAX_WORDS", 450)
CHUNK_OVERLAP_WORDS = env_int("CHUNK_OVERLAP_WORDS", 80)
CHUNK_MIN_WORDS = env_int("CHUNK_MIN_WORDS", 20)
EMBEDDING_BATCH_SIZE = env_int("EMBEDDING_BATCH_SIZE", 8)

# App / upload jobs
UPLOAD_JOB_HISTORY_LIMIT = env_int("UPLOAD_JOB_HISTORY_LIMIT", 100)
DELETE_SAVED_UPLOADS_AFTER_JOB = env_bool("DELETE_SAVED_UPLOADS_AFTER_JOB", False)
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = env_int("APP_PORT", 8080)
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")

NOT_FOUND_TR = "Bu bilgi yuklenen dokumanlarda bulunamadi."
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=getattr(logging, APP_LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("multi-vector-rag-app")

app = FastAPI(title="Multi Vector Database PDF RAG")


class AskRequest(BaseModel):
    question: str


UPLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
UPLOAD_JOB_LOCK = threading.Lock()
MILVUS_LOAD_LOCK = threading.Lock()
MILVUS_LOADED = False


# ============================================================
# COMMON UTILITIES
# ============================================================


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def chunk_batches(items: List[Any], batch_size: int):
    for i in range(0, len(items), max(1, batch_size)):
        yield items[i : i + max(1, batch_size)]


def vector_to_sql(v: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def vector_to_halfvec_sql(v: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in v[:HALFVEC_DIM]) + "]"


def to_oracle_float32_vector(v: List[float]):
    return array("f", [float(x) for x in v])


def auth_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def parse_embedding_response(data: Dict[str, Any], expected_count: int) -> List[List[float]]:
    items = data.get("data")
    if not isinstance(items, list):
        logger.error("Unexpected embedding response: %s", data)
        raise HTTPException(status_code=500, detail="Bad embedding response.")

    if all(isinstance(item, dict) and "index" in item for item in items):
        items = sorted(items, key=lambda item: item["index"])

    embeddings: List[List[float]] = []
    for item in items:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(emb, list):
            raise HTTPException(status_code=500, detail="Bad embedding response.")
        if len(emb) != EMBEDDING_DIM:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding dim mismatch. Expected={EMBEDDING_DIM}, Got={len(emb)}",
            )
        embeddings.append(emb)

    if len(embeddings) != expected_count:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding count mismatch. Expected={expected_count}, Got={len(embeddings)}",
        )
    return embeddings


def request_embeddings(inputs: Any) -> List[List[float]]:
    if not REMOTE_API_URL or not REMOTE_MODEL_NAME:
        raise HTTPException(status_code=500, detail="Embedding API configuration is missing.")

    expected_count = len(inputs) if isinstance(inputs, list) else 1
    payload = {"model": REMOTE_MODEL_NAME, "input": inputs}
    response = requests.post(
        REMOTE_API_URL,
        headers=auth_headers(REMOTE_API_KEY),
        json=payload,
        timeout=EMBEDDING_REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        logger.error("Embedding API error: %s - %s", response.status_code, response.text)
        raise HTTPException(status_code=500, detail="Embedding API error.")
    return parse_embedding_response(response.json(), expected_count)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for batch in chunk_batches(texts, EMBEDDING_BATCH_SIZE):
        inputs: Any = batch[0] if len(batch) == 1 else batch
        embeddings.extend(request_embeddings(inputs))
    return embeddings


def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]


def call_chat_model(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout: Optional[int] = None,
    label: str = "chat",
) -> str:
    if not REMOTE_CHAT_URL or not REMOTE_CHAT_MODEL:
        raise HTTPException(status_code=500, detail="Chat API configuration is missing.")

    payload = {
        "model": REMOTE_CHAT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    started = now()
    logger.info(
        "LLM call started label=%s system_chars=%s user_chars=%s",
        label,
        len(system_prompt),
        len(user_prompt),
    )
    response = requests.post(
        REMOTE_CHAT_URL,
        headers=auth_headers(REMOTE_CHAT_API_KEY),
        json=payload,
        timeout=timeout or CHAT_REQUEST_TIMEOUT,
    )
    logger.info(
        "LLM call finished label=%s duration=%ss status=%s",
        label,
        elapsed(started),
        response.status_code,
    )
    if response.status_code != 200:
        logger.error("Chat API error: %s - %s", response.status_code, response.text)
        raise HTTPException(status_code=500, detail="Chat API error.")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("Unexpected chat response: %s", data)
        raise HTTPException(status_code=500, detail="Bad chat response.") from exc


def ask_llm(question: str, context: str) -> str:
    system_prompt = """
You are a strict document-based RAG assistant.
Use only the provided context.
If the context contains enough information to answer, answer clearly in Turkish.
Keep the answer readable with short paragraphs and bullets where useful.
Do not include a separate sources section; the application renders sources separately.
If the answer is not supported by the context, say exactly:
Bu bilgi yuklenen dokumanlarda bulunamadi.
Do not use outside knowledge and do not guess.
""".strip()
    return call_chat_model(
        system_prompt,
        f"Context:\n{context}\n\nQuestion:\n{question}",
        timeout=CHAT_REQUEST_TIMEOUT,
        label="final_answer",
    )


# ============================================================
# PDF EXTRACTION AND CHUNKING
# ============================================================


def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages: List[str] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n--- PAGE {page_no} ---\n{text}")
    return "\n".join(pages)


def normalize_text_space(text: str) -> str:
    return " ".join((text or "").split())


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def split_text_pages(text: str) -> List[Tuple[Optional[str], str]]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    marker_re = re.compile(r"---\s*PAGE\s+(\d+)\s*---", re.IGNORECASE)
    matches = list(marker_re.finditer(normalized))
    if not matches:
        return [(None, normalized)]

    pages: List[Tuple[Optional[str], str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        pages.append((match.group(1), normalized[start:end].strip()))
    return pages


def split_sentences(text: str) -> List[str]:
    normalized = normalize_text_space(text)
    if not normalized:
        return []
    values = [x.strip() for x in re.split(r"(?<=[.!?])\s+", normalized) if x.strip()]
    return values or [normalized]


def split_words_window(text: str, max_words: int, overlap_words: int = 0) -> List[str]:
    words = normalize_text_space(text).split()
    if not words:
        return []
    overlap_words = max(0, min(overlap_words, max_words - 1))
    step = max(1, max_words - overlap_words)
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    paragraph = normalize_text_space(paragraph)
    if not paragraph:
        return []
    if count_words(paragraph) <= max_words:
        return [paragraph]

    units: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in split_sentences(paragraph):
        sentence_words = count_words(sentence)
        if sentence_words > max_words:
            if current:
                units.append(" ".join(current))
                current = []
                current_words = 0
            units.extend(split_words_window(sentence, max_words, CHUNK_OVERLAP_WORDS))
            continue
        if current and current_words + sentence_words > max_words:
            units.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        units.append(" ".join(current))
    return units


def build_page_chunks(
    page_no: Optional[str],
    units: List[str],
    max_words: int,
    overlap_words: int,
) -> List[str]:
    chunks: List[str] = []
    current_parts: List[str] = []
    current_words = 0
    overlap_words = max(0, min(overlap_words, max_words - 1))

    def flush(carry_overlap: bool) -> None:
        nonlocal current_parts, current_words
        plain = normalize_text_space(" ".join(current_parts))
        if plain:
            chunks.append(f"--- PAGE {page_no} ---\n{plain}" if page_no else plain)
        if carry_overlap and plain and overlap_words:
            overlap_text = " ".join(plain.split()[-overlap_words:])
            current_parts = [overlap_text] if overlap_text else []
            current_words = count_words(overlap_text)
        else:
            current_parts = []
            current_words = 0

    for unit in units:
        unit = normalize_text_space(unit)
        unit_words = count_words(unit)
        if not unit_words:
            continue
        if current_parts and current_words + unit_words > max_words:
            flush(carry_overlap=True)
        if current_words + unit_words > max_words:
            current_parts = []
            current_words = 0
        current_parts.append(unit)
        current_words += unit_words
    flush(carry_overlap=False)
    return chunks


def chunk_text(text: str) -> List[str]:
    raw_chunks: List[str] = []
    for page_no, page_text in split_text_pages(text):
        paragraphs = [normalize_text_space(x) for x in re.split(r"\n\s*\n+", page_text) if x.strip()]
        units: List[str] = []
        for paragraph in paragraphs:
            units.extend(split_long_paragraph(paragraph, CHUNK_MAX_WORDS))
        raw_chunks.extend(
            build_page_chunks(page_no, units, CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)
        )
    return [x for x in raw_chunks if count_words(x) >= CHUNK_MIN_WORDS]


# ============================================================
# QUERY REWRITE, RERANK, CONTEXT
# ============================================================


def rewrite_question_for_search(question: str) -> str:
    if not ENABLE_QUERY_REWRITE:
        return question

    system_prompt = """
Rewrite user questions into concise semantic/vector search queries.
Expand abbreviations, product names, error codes, and likely technical terms when useful.
Return only the rewritten query. Do not answer the question.
""".strip()
    user_prompt = f"Original question:\n{question}\n\nKeep the rewritten query under 40 words."
    try:
        rewritten = call_chat_model(
            system_prompt,
            user_prompt,
            timeout=60,
            label="rewrite",
        )
        rewritten = normalize_text_space(rewritten.strip().strip('"').strip("'"))
        return rewritten or question
    except Exception as exc:
        logger.warning("Question rewrite failed; using original question: %s", exc)
        return question


def prepare_query_embedding(question: str) -> Tuple[str, List[float], float, float]:
    started = now()
    search_query = rewrite_question_for_search(question)
    rewrite_time = elapsed(started)

    started = now()
    q_emb = get_embedding(search_query)
    embed_time = elapsed(started)
    return search_query, q_emb, rewrite_time, embed_time


def retrieval_candidate_limit() -> int:
    return max(TOP_K, RETRIEVAL_CANDIDATE_K)


def sort_chunks_by_similarity(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: float(item.get("similarity") or 0.0),
        reverse=True,
    )[:TOP_K]


def parse_rerank_response(data: Any) -> Dict[int, float]:
    if isinstance(data, dict):
        values = (
            data.get("results")
            or data.get("data")
            or data.get("rankings")
            or data.get("scores")
        )
    else:
        values = data

    if not isinstance(values, list):
        raise ValueError("Rerank response does not contain a result list.")

    scores: Dict[int, float] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index", item.get("id", item.get("document_index")))
        raw_score = item.get(
            "relevance_score",
            item.get("score", item.get("relevance", item.get("similarity"))),
        )
        if raw_index is None or raw_score is None:
            continue
        scores[int(raw_index)] = float(raw_score)

    if not scores:
        raise ValueError("Rerank response did not contain usable scores.")
    return scores


def call_rerank_api(question: str, documents: List[str], top_n: int) -> Dict[int, float]:
    if not RERANK_API_URL or not RERANK_MODEL:
        raise RuntimeError("RERANK_API_URL or RERANK_MODEL is missing.")

    if RERANK_API_STYLE == "tei":
        payload: Dict[str, Any] = {
            "query": question,
            "texts": documents,
            "truncate": True,
        }
        if RERANK_MODEL:
            payload["model"] = RERANK_MODEL
    else:
        payload = {
            "model": RERANK_MODEL,
            "query": question,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

    started = now()
    logger.info(
        "Rerank API call started style=%s candidates=%s chars=%s",
        RERANK_API_STYLE,
        len(documents),
        sum(len(x) for x in documents),
    )
    response = requests.post(
        RERANK_API_URL,
        headers=auth_headers(RERANK_API_KEY),
        json=payload,
        timeout=RERANK_TIMEOUT,
    )
    logger.info(
        "Rerank API call finished duration=%ss status=%s",
        elapsed(started),
        response.status_code,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Rerank API error {response.status_code}: {response.text[:500]}")
    return parse_rerank_response(response.json())


def extract_json_payload(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if not match:
            raise
        return json.loads(match.group(1))


def call_llm_reranker(question: str, documents: List[str]) -> Dict[int, float]:
    candidates = []
    for idx, document in enumerate(documents):
        candidates.append(f"ID: {idx}\nText: {document}")

    system_prompt = """
You are a strict document reranker.
Score every candidate for usefulness in answering the question.
Return only JSON in this form: [{"id": 0, "score": 0.0}]
Scores must be between 0 and 1.
""".strip()
    response_text = call_chat_model(
        system_prompt,
        f"Question:\n{question}\n\nCandidates:\n" + "\n\n---\n\n".join(candidates),
        timeout=RERANK_TIMEOUT,
        label="llm_rerank",
    )
    return parse_rerank_response(extract_json_payload(response_text))


def rerank_chunks(question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    fallback = sort_chunks_by_similarity(candidates)
    if RERANK_MODE == "none":
        return fallback

    limited = candidates[: max(TOP_K, RERANK_MAX_CANDIDATES)]
    documents = [
        normalize_text_space(str(item.get("content") or ""))[:RERANK_CHUNK_MAX_CHARS]
        for item in limited
    ]

    try:
        if RERANK_MODE == "api":
            score_by_id = call_rerank_api(question, documents, TOP_K)
        elif RERANK_MODE == "llm":
            score_by_id = call_llm_reranker(question, documents)
        else:
            logger.warning("Unknown RERANK_MODE=%s; using vector scores", RERANK_MODE)
            return fallback
    except Exception as exc:
        logger.warning("Rerank failed; fallback=%s error=%s", RERANK_FALLBACK, exc)
        if RERANK_FALLBACK == "llm" and RERANK_MODE != "llm":
            try:
                score_by_id = call_llm_reranker(question, documents)
            except Exception as llm_exc:
                logger.warning("LLM rerank fallback failed: %s", llm_exc)
                return fallback
        else:
            return fallback

    reranked: List[Dict[str, Any]] = []
    for idx, item in enumerate(limited):
        copied = dict(item)
        copied["rerank_score"] = score_by_id.get(idx, float(item.get("similarity") or 0.0))
        reranked.append(copied)

    return sorted(
        reranked,
        key=lambda item: (
            float(item.get("rerank_score") or 0.0),
            float(item.get("similarity") or 0.0),
        ),
        reverse=True,
    )[:TOP_K]


def build_context_from_items(items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    context = ""
    sources: List[Dict[str, Any]] = []
    for item in items:
        piece = (
            f"\nSOURCE: {item['file_name']} | CHUNK: {item['chunk_index']} "
            f"| SIMILARITY: {float(item['similarity']):.4f}\n"
            f"{item['content']}\n"
        )
        if len(context) + len(piece) > MAX_CONTEXT_LENGTH:
            break
        context += piece
        source = {
            "file_name": item["file_name"],
            "chunk_index": item["chunk_index"],
            "similarity": float(item["similarity"]),
        }
        if "rerank_score" in item:
            source["rerank_score"] = float(item["rerank_score"])
        sources.append(source)
    return context, sources


def empty_result(
    total_start: float,
    *,
    rewrite_time: float = 0,
    embed_time: float = 0,
    retrieval_time: float = 0,
    rerank_time: float = 0,
) -> Dict[str, Any]:
    return {
        "answer": NOT_FOUND_TR,
        "sources": [],
        "timings": {
            "rewrite": rewrite_time,
            "embedding": embed_time,
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "llm": 0,
            "total": elapsed(total_start),
        },
    }


def run_ask_pipeline(
    question: str,
    retrieval_fn: Callable[[List[float], int], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    total_start = now()
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    _search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    candidate_limit = retrieval_candidate_limit()

    started = now()
    items = retrieval_fn(q_emb, candidate_limit)
    retrieval_time = elapsed(started)

    if not items or float(items[0].get("similarity") or 0.0) < SIMILARITY_THRESHOLD:
        return empty_result(
            total_start,
            rewrite_time=rewrite_time,
            embed_time=embed_time,
            retrieval_time=retrieval_time,
        )

    started = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(started)

    context, sources = build_context_from_items(ranked_items)
    if not context.strip():
        return empty_result(
            total_start,
            rewrite_time=rewrite_time,
            embed_time=embed_time,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
        )

    started = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(started)

    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "rewrite": rewrite_time,
            "embedding": embed_time,
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "llm": llm_time,
            "total": elapsed(total_start),
        },
    }


# ============================================================
# UPLOAD JOBS
# ============================================================


def copy_upload_job(job: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(job)
    copied["files"] = [dict(x) for x in job.get("files", [])]
    copied["timings"] = dict(job.get("timings", {}))
    if job.get("result"):
        copied["result"] = dict(job["result"])
    return copied


def prune_upload_jobs_locked() -> None:
    if len(UPLOAD_JOBS) <= UPLOAD_JOB_HISTORY_LIMIT:
        return
    finished = [j for j in UPLOAD_JOBS.values() if j.get("status") in {"success", "failed"}]
    finished.sort(key=lambda j: j.get("updated_at", 0))
    for job in finished:
        if len(UPLOAD_JOBS) <= UPLOAD_JOB_HISTORY_LIMIT:
            break
        UPLOAD_JOBS.pop(job["job_id"], None)


def create_upload_job(backend: str, saved_files: List[Dict[str, str]]) -> str:
    job_id = str(uuid.uuid4())
    timestamp = time.time()
    with UPLOAD_JOB_LOCK:
        UPLOAD_JOBS[job_id] = {
            "job_id": job_id,
            "backend": backend,
            "status": "queued",
            "message": "Queued.",
            "current_file": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "files_total": len(saved_files),
            "files_done": 0,
            "chunks_total": 0,
            "chunks_done": 0,
            "inserted_chunks": 0,
            "files": [
                {"file": item["original_name"], "status": "queued", "chunks": 0}
                for item in saved_files
            ],
            "timings": {},
            "error": None,
            "result": None,
        }
        prune_upload_jobs_locked()
    return job_id


def get_upload_job(job_id: str) -> Optional[Dict[str, Any]]:
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        return copy_upload_job(job) if job else None


def update_upload_job(job_id: str, **updates: Any) -> None:
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def increment_upload_job(job_id: str, message: Optional[str] = None, **increments: int) -> None:
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return
        for key, amount in increments.items():
            job[key] = job.get(key, 0) + amount
        if message is not None:
            job["message"] = message
        job["updated_at"] = time.time()


def record_upload_file_result(job_id: str, file_name: str, status: str, chunks: int) -> None:
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return
        for item in job.get("files", []):
            if item.get("file") == file_name and item.get("status") != "ok":
                item.update({"status": status, "chunks": chunks})
                break
        job["updated_at"] = time.time()


def finish_upload_job(job_id: str, result: Dict[str, Any]) -> None:
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.update(
            {
                "status": "success",
                "message": "Completed.",
                "inserted_chunks": result.get("inserted_chunks", 0),
                "files": result.get("files", job.get("files", [])),
                "timings": result.get("timings", {}),
                "result": result,
                "updated_at": time.time(),
            }
        )


def fail_upload_job(job_id: str, exc: Exception) -> None:
    detail = getattr(exc, "detail", None) or str(exc)
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.update(
            {
                "status": "failed",
                "message": "Failed.",
                "error": detail,
                "updated_at": time.time(),
            }
        )


async def save_uploaded_pdfs(files: List[UploadFile]) -> List[Dict[str, str]]:
    saved: List[Dict[str, str]] = []
    for uploaded_file in files:
        original_name = os.path.basename(uploaded_file.filename or "")
        if not original_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        path = UPLOAD_DIR / f"{uuid.uuid4()}_{original_name}"
        path.write_bytes(await uploaded_file.read())
        saved.append({"original_name": original_name, "file_path": str(path)})
    return saved


def embed_chunks_for_upload(
    job_id: str,
    file_name: str,
    chunks: List[str],
) -> Tuple[List[List[float]], float]:
    embeddings: List[List[float]] = []
    embedding_time = 0.0
    for batch in chunk_batches(chunks, EMBEDDING_BATCH_SIZE):
        started = now()
        inputs: Any = batch[0] if len(batch) == 1 else batch
        batch_embeddings = request_embeddings(inputs)
        embedding_time += elapsed(started)
        embeddings.extend(batch_embeddings)
        increment_upload_job(
            job_id,
            chunks_done=len(batch),
            message=f"Embedding {file_name}: {len(embeddings)}/{len(chunks)} chunks.",
        )
    return embeddings, round(embedding_time, 4)


def run_upload_job(
    job_id: str,
    backend: str,
    saved_files: List[Dict[str, str]],
    process_func: Callable[[str, List[Dict[str, str]]], Dict[str, Any]],
) -> None:
    update_upload_job(job_id, status="running", message=f"{backend} upload started.")
    try:
        result = process_func(job_id, saved_files)
        finish_upload_job(job_id, result)
    except Exception as exc:
        logger.exception("Upload job failed job_id=%s backend=%s", job_id, backend)
        fail_upload_job(job_id, exc)
    finally:
        if DELETE_SAVED_UPLOADS_AFTER_JOB:
            for item in saved_files:
                try:
                    Path(item["file_path"]).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Could not delete temporary upload %s", item["file_path"])


async def enqueue_upload_job(
    backend: str,
    background_tasks: BackgroundTasks,
    process_func: Callable[[str, List[Dict[str, str]]], Dict[str, Any]],
    files: List[UploadFile],
) -> Dict[str, Any]:
    saved_files = await save_uploaded_pdfs(files)
    job_id = create_upload_job(backend, saved_files)
    background_tasks.add_task(run_upload_job, job_id, backend, saved_files, process_func)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job_url": f"/upload-jobs/{job_id}",
        "message": "Upload job queued.",
        "files": [item["original_name"] for item in saved_files],
    }


# ============================================================
# CONNECTIONS AND COLLECTION/INDEX ENSURE FUNCTIONS
# ============================================================


def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def connect_milvus() -> None:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=MILVUS_OPERATION_TIMEOUT,
    )


def milvus_progress(job_id: str, detailed: str, concise: str) -> None:
    """Keep the normal UI clean unless detailed Milvus progress is enabled."""
    update_upload_job(
        job_id,
        message=detailed if MILVUS_DETAILED_PROGRESS else concise,
    )


def milvus_index_params() -> Dict[str, Any]:
    """Return index parameters matching the configured Milvus index type."""
    if MILVUS_INDEX_TYPE == "HNSW":
        return {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": MILVUS_HNSW_M,
                "efConstruction": MILVUS_HNSW_EF_CONSTRUCTION,
            },
        }
    if MILVUS_INDEX_TYPE == "IVF_FLAT":
        return {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": MILVUS_IVF_NLIST},
        }
    return {
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {},
    }


def milvus_collection_has_index(collection: Collection) -> bool:
    try:
        return bool(collection.indexes)
    except Exception as exc:
        logger.warning("Could not inspect Milvus indexes: %s", exc)
        return False


def get_milvus_collection() -> Collection:
    """
    Connect to Milvus and create only the collection schema when it is missing.

    The vector index is deliberately not created here. On some standalone/version
    combinations create_index() on a brand-new empty collection can wait for a long
    time. The first upload inserts the vectors, flushes once, and then creates the
    index via ensure_milvus_index().
    """
    global MILVUS_LOADED

    logger.info(
        "Milvus connect starting host=%s port=%s timeout=%s",
        MILVUS_HOST,
        MILVUS_PORT,
        MILVUS_OPERATION_TIMEOUT,
    )
    connect_milvus()
    logger.info("Milvus connect finished")

    logger.info("Milvus collection existence check starting name=%s", MILVUS_COLLECTION)
    exists = utility.has_collection(MILVUS_COLLECTION)
    logger.info(
        "Milvus collection existence check finished name=%s exists=%s",
        MILVUS_COLLECTION,
        exists,
    )

    if exists:
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    collection_kwargs = {
        "name": MILVUS_COLLECTION,
        "schema": CollectionSchema(
            fields,
            description="PDF RAG collection",
            enable_dynamic_field=False,
        ),
        "using": "default",
        "shards_num": 1,
    }

    logger.info(
        "Milvus collection creation starting name=%s dim=%s",
        MILVUS_COLLECTION,
        EMBEDDING_DIM,
    )
    try:
        collection = Collection(
            **collection_kwargs,
            consistency_level=MILVUS_CONSISTENCY_LEVEL,
        )
    except TypeError:
        collection = Collection(**collection_kwargs)

    MILVUS_LOADED = False
    logger.info(
        "Milvus collection created name=%s; index deferred until after first insert",
        MILVUS_COLLECTION,
    )
    return collection


def ensure_milvus_index(collection: Collection, job_id: str = None) -> None:
    """Create the configured vector index once, after vectors have been inserted."""
    global MILVUS_LOADED

    if milvus_collection_has_index(collection):
        logger.info("Milvus vector index already exists collection=%s", MILVUS_COLLECTION)
        return

    if job_id:
        milvus_progress(
            job_id,
            detailed=(
                f"Flushing first Milvus data before creating {MILVUS_INDEX_TYPE} index."
            ),
            concise="Preparing Milvus index.",
        )

    logger.info(
        "Milvus pre-index flush starting collection=%s timeout=%s",
        MILVUS_COLLECTION,
        MILVUS_FLUSH_TIMEOUT,
    )
    collection.flush(timeout=MILVUS_FLUSH_TIMEOUT)
    logger.info("Milvus pre-index flush finished collection=%s", MILVUS_COLLECTION)

    if job_id:
        milvus_progress(
            job_id,
            detailed=(
                f"Creating Milvus {MILVUS_INDEX_TYPE} index on embedding field."
            ),
            concise="Creating Milvus index.",
        )

    params = milvus_index_params()
    logger.info(
        "Milvus index creation starting collection=%s type=%s params=%s timeout=%s",
        MILVUS_COLLECTION,
        MILVUS_INDEX_TYPE,
        params,
        MILVUS_INDEX_TIMEOUT,
    )
    collection.create_index(
        field_name="embedding",
        index_params=params,
        timeout=MILVUS_INDEX_TIMEOUT,
    )
    MILVUS_LOADED = False
    logger.info(
        "Milvus index creation finished collection=%s type=%s",
        MILVUS_COLLECTION,
        MILVUS_INDEX_TYPE,
    )

def ensure_milvus_loaded(collection: Collection) -> None:
    global MILVUS_LOADED
    if MILVUS_LOADED:
        return
    with MILVUS_LOAD_LOCK:
        if not MILVUS_LOADED:
            collection.load(timeout=MILVUS_OPERATION_TIMEOUT)
            MILVUS_LOADED = True


def milvus_search_with_consistency(collection: Collection, **kwargs):
    kwargs["consistency_level"] = MILVUS_CONSISTENCY_LEVEL
    try:
        return collection.search(**kwargs)
    except TypeError as exc:
        if "consistency_level" not in str(exc):
            raise
        kwargs.pop("consistency_level", None)
        return collection.search(**kwargs)


def milvus_query_with_consistency(collection: Collection, **kwargs):
    kwargs["consistency_level"] = MILVUS_CONSISTENCY_LEVEL
    try:
        return collection.query(**kwargs)
    except TypeError as exc:
        if "consistency_level" not in str(exc):
            raise
        kwargs.pop("consistency_level", None)
        return collection.query(**kwargs)


def get_oracle_connection():
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)


def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
    )


def ensure_qdrant_collection() -> QdrantClient:
    client = get_qdrant_client()
    names = [x.name for x in client.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection=%s", QDRANT_COLLECTION)
    return client


def get_es_client() -> Elasticsearch:
    hosts = [x.strip() for x in ES_URL.split(",") if x.strip()]
    use_https = any(x.lower().startswith("https://") for x in hosts)
    kwargs: Dict[str, Any] = {"request_timeout": ES_REQUEST_TIMEOUT}
    if use_https:
        kwargs["verify_certs"] = ES_VERIFY_CERTS
    if use_https and ES_CA_CERTS:
        kwargs["ca_certs"] = ES_CA_CERTS
    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USERNAME and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)
    return Elasticsearch(hosts, **kwargs)


def ensure_es_index() -> Elasticsearch:
    if EMBEDDING_DIM > 4096:
        raise HTTPException(status_code=500, detail="Elasticsearch requires EMBEDDING_DIM <= 4096.")
    client = get_es_client()
    if not bool(client.indices.exists(index=ES_INDEX)):
        client.indices.create(
            index=ES_INDEX,
            mappings={
                "properties": {
                    "file_name": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )
        logger.info("Created Elasticsearch index=%s", ES_INDEX)
    return client


# ============================================================
# BACKEND UPLOAD IMPLEMENTATIONS
# ============================================================


def pg_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = embed_time = db_time = 0.0
    inserted_total = 0
    results = []
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for saved in saved_files:
                    file_name, file_path = saved["original_name"], saved["file_path"]
                    update_upload_job(job_id, current_file=file_name, message=f"Extracting {file_name}.")
                    started = now()
                    chunks = chunk_text(extract_pdf_text(file_path))
                    extract_time += elapsed(started)
                    increment_upload_job(job_id, chunks_total=len(chunks))
                    embeddings, batch_time = embed_chunks_for_upload(job_id, file_name, chunks)
                    embed_time += batch_time
                    rows = [
                        (file_name, idx, chunk, vector_to_sql(emb), vector_to_halfvec_sql(emb))
                        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ]
                    started = now()
                    update_upload_job(job_id, message=f"Inserting {file_name} into PGVector.")
                    if rows:
                        execute_values(
                            cur,
                            f"INSERT INTO {PG_TABLE} "
                            "(file_name, chunk_index, content, embedding, embedding_half) VALUES %s",
                            rows,
                            template=(
                                f"(%s, %s, %s, %s::vector({EMBEDDING_DIM}), "
                                f"%s::halfvec({HALFVEC_DIM}))"
                            ),
                        )
                    db_time += elapsed(started)
                    inserted_total += len(rows)
                    increment_upload_job(job_id, files_done=1, inserted_chunks=len(rows))
                    record_upload_file_result(job_id, file_name, "ok", len(rows))
                    results.append({"file": file_name, "status": "ok", "chunks": len(rows)})
        return {
            "status": "success",
            "inserted_chunks": inserted_total,
            "files": results,
            "timings": {
                "extract_chunk": round(extract_time, 4),
                "embedding": round(embed_time, 4),
                "db_insert": round(db_time, 4),
                "total": elapsed(total_start),
            },
        }
    finally:
        conn.close()


def milvus_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = embed_time = db_time = 0.0
    inserted_total = 0
    results = []

    milvus_progress(
        job_id,
        detailed=f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}.",
        concise="Preparing Milvus upload.",
    )
    collection = get_milvus_collection()

    for saved in saved_files:
        file_name, file_path = saved["original_name"], saved["file_path"]
        update_upload_job(job_id, current_file=file_name, message=f"Extracting {file_name}.")

        started = now()
        chunks = chunk_text(extract_pdf_text(file_path))
        extract_time += elapsed(started)
        increment_upload_job(job_id, chunks_total=len(chunks))

        embeddings, batch_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_time
        rows = [
            {
                "file_name": file_name,
                "chunk_index": idx,
                "content": chunk[:8000],
                "embedding": emb,
            }
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        started = now()
        batch_size = max(1, MILVUS_INSERT_BATCH_SIZE)
        batch_total = max(1, (len(rows) + batch_size - 1) // batch_size)

        for batch_no, batch in enumerate(chunk_batches(rows, batch_size), start=1):
            milvus_progress(
                job_id,
                detailed=(
                    f"Inserting {file_name} into Milvus: "
                    f"batch {batch_no}/{batch_total}, rows={len(batch)}."
                ),
                concise=f"Inserting {file_name} into Milvus.",
            )
            logger.info(
                "Milvus insert starting file=%s batch=%s/%s rows=%s",
                file_name,
                batch_no,
                batch_total,
                len(batch),
            )

            result = collection.insert(
                [
                    [x["file_name"] for x in batch],
                    [x["chunk_index"] for x in batch],
                    [x["content"] for x in batch],
                    [x["embedding"] for x in batch],
                ],
                timeout=MILVUS_OPERATION_TIMEOUT,
            )

            logger.info(
                "Milvus insert finished file=%s batch=%s/%s insert_count=%s",
                file_name,
                batch_no,
                batch_total,
                getattr(result, "insert_count", None),
            )

        # Do not create/load the index inside get_milvus_collection().
        # On the first upload the index is created once after all files are inserted.
        if MILVUS_FLUSH_AFTER_UPLOAD and rows and milvus_collection_has_index(collection):
            milvus_progress(
                job_id,
                detailed=f"Flushing {file_name} in Milvus.",
                concise=f"Finalizing {file_name} in Milvus.",
            )
            logger.info(
                "Milvus flush starting file=%s timeout=%s",
                file_name,
                MILVUS_FLUSH_TIMEOUT,
            )
            collection.flush(timeout=MILVUS_FLUSH_TIMEOUT)
            logger.info("Milvus flush finished file=%s", file_name)

        db_time += elapsed(started)
        inserted_total += len(rows)
        increment_upload_job(
            job_id,
            files_done=1,
            inserted_chunks=len(rows),
            message=f"Finished {file_name}.",
        )
        record_upload_file_result(job_id, file_name, "ok", len(rows))
        results.append({"file": file_name, "status": "ok", "chunks": len(rows)})

    # Required first-run sequence: create collection -> insert -> flush -> create index.
    if inserted_total > 0:
        index_started = now()
        ensure_milvus_index(collection, job_id=job_id)
        db_time += elapsed(index_started)

    return {
        "status": "success",
        "inserted_chunks": inserted_total,
        "files": results,
        "timings": {
            "extract_chunk": round(extract_time, 4),
            "embedding": round(embed_time, 4),
            "db_insert": round(db_time, 4),
            "total": elapsed(total_start),
        },
    }


def oracle_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = embed_time = db_time = 0.0
    inserted_total = 0
    results = []
    conn = get_oracle_connection()
    try:
        cur = conn.cursor()
        cur.setinputsizes(
            file_name=oracledb.DB_TYPE_VARCHAR,
            chunk_index=oracledb.DB_TYPE_NUMBER,
            content=oracledb.DB_TYPE_CLOB,
        )
        for saved in saved_files:
            file_name, file_path = saved["original_name"], saved["file_path"]
            update_upload_job(job_id, current_file=file_name, message=f"Extracting {file_name}.")
            started = now()
            chunks = chunk_text(extract_pdf_text(file_path))
            extract_time += elapsed(started)
            increment_upload_job(job_id, chunks_total=len(chunks))
            embeddings, batch_time = embed_chunks_for_upload(job_id, file_name, chunks)
            embed_time += batch_time
            rows = [
                {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": to_oracle_float32_vector(emb),
                }
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
            started = now()
            update_upload_job(job_id, message=f"Inserting {file_name} into Oracle.")
            if rows:
                cur.executemany(
                    f"INSERT INTO {ORACLE_TABLE} "
                    "(file_name, chunk_index, content, embedding) "
                    "VALUES (:file_name, :chunk_index, :content, :embedding)",
                    rows,
                )
            conn.commit()
            db_time += elapsed(started)
            inserted_total += len(rows)
            increment_upload_job(job_id, files_done=1, inserted_chunks=len(rows))
            record_upload_file_result(job_id, file_name, "ok", len(rows))
            results.append({"file": file_name, "status": "ok", "chunks": len(rows)})
        return {
            "status": "success",
            "inserted_chunks": inserted_total,
            "files": results,
            "timings": {
                "extract_chunk": round(extract_time, 4),
                "embedding": round(embed_time, 4),
                "db_insert": round(db_time, 4),
                "total": elapsed(total_start),
            },
        }
    finally:
        conn.close()


def qdrant_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = embed_time = db_time = 0.0
    inserted_total = 0
    results = []
    client = ensure_qdrant_collection()

    for saved in saved_files:
        file_name, file_path = saved["original_name"], saved["file_path"]
        update_upload_job(job_id, current_file=file_name, message=f"Extracting {file_name}.")
        started = now()
        chunks = chunk_text(extract_pdf_text(file_path))
        extract_time += elapsed(started)
        increment_upload_job(job_id, chunks_total=len(chunks))
        embeddings, batch_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_time
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"file_name": file_name, "chunk_index": idx, "content": chunk},
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        started = now()
        update_upload_job(job_id, message=f"Inserting {file_name} into Qdrant.")
        for batch in chunk_batches(points, QDRANT_INSERT_BATCH_SIZE):
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch,
                wait=QDRANT_WAIT,
            )
        db_time += elapsed(started)
        inserted_total += len(points)
        increment_upload_job(job_id, files_done=1, inserted_chunks=len(points))
        record_upload_file_result(job_id, file_name, "ok", len(points))
        results.append({"file": file_name, "status": "ok", "chunks": len(points)})

    return {
        "status": "success",
        "inserted_chunks": inserted_total,
        "files": results,
        "timings": {
            "extract_chunk": round(extract_time, 4),
            "embedding": round(embed_time, 4),
            "db_insert": round(db_time, 4),
            "total": elapsed(total_start),
        },
    }


def elasticsearch_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = embed_time = db_time = 0.0
    inserted_total = 0
    results = []
    client = ensure_es_index()

    for saved in saved_files:
        file_name, file_path = saved["original_name"], saved["file_path"]
        update_upload_job(job_id, current_file=file_name, message=f"Extracting {file_name}.")
        started = now()
        chunks = chunk_text(extract_pdf_text(file_path))
        extract_time += elapsed(started)
        increment_upload_job(job_id, chunks_total=len(chunks))
        embeddings, batch_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_time
        actions = [
            {
                "_index": ES_INDEX,
                "_id": str(uuid.uuid4()),
                "_source": {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": emb,
                },
            }
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        started = now()
        update_upload_job(job_id, message=f"Inserting {file_name} into Elasticsearch.")
        for batch in chunk_batches(actions, ES_INSERT_BATCH_SIZE):
            helpers.bulk(client, batch, request_timeout=ES_REQUEST_TIMEOUT)
        if actions:
            client.indices.refresh(index=ES_INDEX)
        db_time += elapsed(started)
        inserted_total += len(actions)
        increment_upload_job(job_id, files_done=1, inserted_chunks=len(actions))
        record_upload_file_result(job_id, file_name, "ok", len(actions))
        results.append({"file": file_name, "status": "ok", "chunks": len(actions)})

    return {
        "status": "success",
        "inserted_chunks": inserted_total,
        "files": results,
        "timings": {
            "extract_chunk": round(extract_time, 4),
            "embedding": round(embed_time, 4),
            "db_insert": round(db_time, 4),
            "total": elapsed(total_start),
        },
    }


# ============================================================
# BACKEND RETRIEVAL IMPLEMENTATIONS
# ============================================================


def pg_retrieve(q_emb: List[float], limit: int) -> List[Dict[str, Any]]:
    q_half = vector_to_halfvec_sql(q_emb)
    q_full = vector_to_sql(q_emb)
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL ivfflat.probes = {max(1, PG_IVFFLAT_PROBES)}")
                cur.execute(
                    f"""
                    WITH candidates AS MATERIALIZED (
                        SELECT id, file_name, chunk_index, content, embedding
                        FROM {PG_TABLE}
                        WHERE embedding_half IS NOT NULL
                        ORDER BY embedding_half <=> %s::halfvec({HALFVEC_DIM})
                        LIMIT %s
                    )
                    SELECT file_name, chunk_index, content,
                           1 - (embedding <=> %s::vector({EMBEDDING_DIM})) AS similarity
                    FROM candidates
                    ORDER BY embedding <=> %s::vector({EMBEDDING_DIM})
                    LIMIT %s
                    """,
                    (
                        q_half,
                        max(PG_CANDIDATE_K, limit),
                        q_full,
                        q_full,
                        limit,
                    ),
                )
                rows = cur.fetchall()
        return [
            {
                "file_name": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def milvus_retrieve(q_emb: List[float], limit: int) -> List[Dict[str, Any]]:
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        return []

    collection = Collection(MILVUS_COLLECTION)
    if not milvus_collection_has_index(collection):
        logger.warning(
            "Milvus collection exists without an index; upload a PDF or run the create_index diagnostic action."
        )
        return []

    ensure_milvus_loaded(collection)
    if MILVUS_INDEX_TYPE == "HNSW":
        params = {"metric_type": "COSINE", "params": {"ef": MILVUS_HNSW_EF}}
    elif MILVUS_INDEX_TYPE == "IVF_FLAT":
        params = {"metric_type": "COSINE", "params": {"nprobe": MILVUS_IVF_NPROBE}}
    else:
        params = {"metric_type": "COSINE", "params": {}}
    result = milvus_search_with_consistency(
        collection,
        data=[q_emb],
        anns_field="embedding",
        param=params,
        limit=limit,
        output_fields=["file_name", "chunk_index", "content"],
        timeout=MILVUS_OPERATION_TIMEOUT,
    )
    items: List[Dict[str, Any]] = []
    for hit in result[0]:
        items.append(
            {
                "file_name": hit.entity.get("file_name"),
                "chunk_index": hit.entity.get("chunk_index"),
                "content": hit.entity.get("content"),
                "similarity": float(hit.score),
            }
        )
    return items


def oracle_retrieve(q_emb: List[float], limit: int) -> List[Dict[str, Any]]:
    q_vec = to_oracle_float32_vector(q_emb)
    if ORACLE_USE_APPROX_SEARCH:
        fetch_clause = f"FETCH FIRST {limit} ROWS ONLY WITH TARGET ACCURACY {ORACLE_TARGET_ACCURACY}"
    else:
        fetch_clause = f"FETCH EXACT FIRST {limit} ROWS ONLY"

    conn = get_oracle_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT file_name, chunk_index, content,
                   1 - VECTOR_DISTANCE(embedding, :q_vec, COSINE) AS similarity
            FROM {ORACLE_TABLE}
            ORDER BY VECTOR_DISTANCE(embedding, :q_vec, COSINE)
            {fetch_clause}
            """,
            q_vec=q_vec,
        )
        rows_raw = cur.fetchall()
        rows = []
        for row in rows_raw:
            content = row[2].read() if hasattr(row[2], "read") else row[2]
            rows.append(
                {
                    "file_name": row[0],
                    "chunk_index": row[1],
                    "content": content,
                    "similarity": float(row[3]),
                }
            )
        return rows
    finally:
        conn.close()


def qdrant_retrieve(q_emb: List[float], limit: int) -> List[Dict[str, Any]]:
    client = ensure_qdrant_collection()
    query_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_emb,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    items = []
    for hit in query_result.points:
        payload = hit.payload or {}
        items.append(
            {
                "file_name": payload.get("file_name"),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content"),
                "similarity": float(hit.score),
            }
        )
    return items


def elastic_score_to_cosine(score: float) -> float:
    return max(-1.0, min(1.0, (float(score) * 2.0) - 1.0))


def elasticsearch_retrieve(q_emb: List[float], limit: int) -> List[Dict[str, Any]]:
    client = ensure_es_index()
    response = client.search(
        index=ES_INDEX,
        knn={
            "field": "embedding",
            "query_vector": q_emb,
            "k": limit,
            "num_candidates": max(ES_NUM_CANDIDATES, limit),
        },
        source=["file_name", "chunk_index", "content"],
        size=limit,
    )
    items = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        items.append(
            {
                "file_name": source.get("file_name"),
                "chunk_index": source.get("chunk_index"),
                "content": source.get("content"),
                "similarity": elastic_score_to_cosine(hit.get("_score", 0)),
            }
        )
    return items


# ============================================================
# UI
# ============================================================


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi Vector RAG</title>
  <style>
    body { margin:0; font-family:Arial,sans-serif; background:#f4f4f5; color:#111827; }
    .container { max-width:1080px; margin:28px auto; padding:18px; }
    .card { background:#fff; border-radius:16px; padding:22px; box-shadow:0 8px 28px rgba(0,0,0,.08); margin-bottom:18px; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
    .tab { padding:10px 15px; border:1px solid #d1d5db; border-radius:9px; cursor:pointer; font-weight:700; background:#f9fafb; }
    .tab.active { color:#fff; background:#111827; }
    .upload { border:2px dashed #9ca3af; border-radius:12px; padding:20px; text-align:center; }
    button { border:0; border-radius:9px; padding:11px 15px; background:#111827; color:#fff; font-weight:700; cursor:pointer; margin:7px 3px 0; }
    button.secondary { background:#fff; color:#111827; border:1px solid #d1d5db; }
    button.warning { background:#92400e; }
    textarea { width:100%; min-height:90px; border:1px solid #d1d5db; border-radius:10px; padding:12px; box-sizing:border-box; }
    .row { display:flex; gap:10px; align-items:flex-start; }
    .row textarea { flex:1; }
    .chat { min-height:280px; max-height:540px; overflow-y:auto; background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:15px; }
    .msg { padding:13px; border-radius:12px; margin-bottom:10px; line-height:1.5; white-space:pre-wrap; }
    .user { background:#e0f2fe; margin-left:70px; }
    .bot { background:#fff; border:1px solid #e5e7eb; margin-right:70px; }
    .bot.formatted { white-space:normal; }
    .section { margin-top:13px; padding-top:11px; border-top:1px solid #e5e7eb; }
    .label { color:#6b7280; font-size:12px; font-weight:700; text-transform:uppercase; margin-bottom:6px; }
    .source { display:grid; grid-template-columns:28px 1fr auto; gap:8px; padding:8px; margin-top:7px; border:1px solid #e5e7eb; border-radius:8px; background:#f9fafb; }
    .rank { width:24px; height:24px; border-radius:50%; background:#111827; color:#fff; display:flex; align-items:center; justify-content:center; font-size:12px; }
    .muted { color:#6b7280; font-size:12px; }
    .chips { display:flex; gap:7px; flex-wrap:wrap; }
    .chip { padding:6px 8px; border:1px solid #e5e7eb; border-radius:8px; background:#f9fafb; font-size:12px; }
    .status { margin-top:10px; color:#6b7280; white-space:pre-wrap; font-size:14px; }
    .hidden { display:none !important; }
    .diag { margin-top:14px; text-align:left; border:1px solid #d1d5db; border-radius:10px; background:#f9fafb; padding:13px; }
    .diag-head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .diag-actions { display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }
    .diag-output { max-height:300px; overflow:auto; white-space:pre-wrap; background:#111827; color:#e5e7eb; border-radius:8px; padding:11px; font:12px/1.45 monospace; margin-top:10px; }
    table { width:100%; border-collapse:collapse; margin-top:12px; }
    th,td { padding:8px; border-bottom:1px solid #e5e7eb; text-align:left; }
  </style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>PDF RAG Chat</h1>
    <div class="tabs">
      <div id="pgTab" class="tab active" onclick="setBackend('pg')">PGVector</div>
      <div id="milvusTab" class="tab" onclick="setBackend('milvus')">Milvus</div>
      <div id="oracleTab" class="tab" onclick="setBackend('oracle')">Oracle 26ai</div>
      <div id="qdrantTab" class="tab" onclick="setBackend('qdrant')">Qdrant</div>
      <div id="elasticTab" class="tab" onclick="setBackend('elasticsearch')">Elasticsearch</div>
    </div>
    <div class="upload">
      <strong id="backendText">Active backend: PGVector</strong><br />
      <input id="pdfFiles" type="file" accept="application/pdf" multiple /><br />
      <button onclick="uploadPdfs()">Upload PDFs</button>
      <button onclick="listDocs()">List PDFs</button>
      <button onclick="clearDocs()">Clear Active DB</button>
      <button id="milvusToolsButton" class="secondary hidden" onclick="toggleMilvusTools()">Milvus Controls</button>
      <div id="uploadStatus" class="status"></div>

      <div id="milvusDiagnosticsPanel" class="diag hidden">
        <div class="diag-head">
          <div>
            <strong>Milvus diagnostics</strong>
            <div class="muted">This panel exists only when MILVUS_DIAGNOSTICS_ENABLED=true.</div>
          </div>
          <button class="secondary" onclick="toggleMilvusTools()">Close</button>
        </div>
        <div class="diag-actions">
          <button class="secondary" onclick="milvusDiagnosticStatus()">Status</button>
          <button class="secondary" onclick="milvusDiagnosticAction('reconnect')">Reconnect</button>
          <button class="secondary" onclick="milvusDiagnosticAction('load')">Load</button>
          <button class="secondary" onclick="milvusDiagnosticAction('release')">Release</button>
          <button class="warning" onclick="milvusDiagnosticAction('flush')">Manual Flush</button>
          <button class="secondary" onclick="milvusDiagnosticAction('create_index')">Create Index</button>
        </div>
        <pre id="milvusDiagnosticsOutput" class="diag-output">No diagnostic request yet.</pre>
      </div>
    </div>
    <div id="docsBox"></div>
  </div>
  <div class="card">
    <div id="chatBox" class="chat"><div class="msg bot">Upload PDFs, then ask a question.</div></div><br />
    <div class="row"><textarea id="question" placeholder="Ask a question..."></textarea><button onclick="askQuestion()">Ask</button></div>
  </div>
</div>
<script>
let backend="pg";
let activeUploadJobId=null;
let milvusDiagnosticsEnabled=false;
let milvusToolsOpen=false;
const names={pg:"PGVector",milvus:"Milvus",oracle:"Oracle 26ai",qdrant:"Qdrant",elasticsearch:"Elasticsearch"};

async function loadAppConfig(){
  try{
    const response=await fetch("/app-config");
    const data=await response.json();
    milvusDiagnosticsEnabled=Boolean(data.milvus_diagnostics_enabled);
    updateMilvusToolsVisibility();
  }catch(error){
    milvusDiagnosticsEnabled=false;
    updateMilvusToolsVisibility();
  }
}

function updateMilvusToolsVisibility(){
  const button=document.getElementById("milvusToolsButton");
  const panel=document.getElementById("milvusDiagnosticsPanel");
  const shouldOffer=backend==="milvus"&&milvusDiagnosticsEnabled;
  button.classList.toggle("hidden",!shouldOffer);
  if(!shouldOffer){
    milvusToolsOpen=false;
    panel.classList.add("hidden");
  }else{
    panel.classList.toggle("hidden",!milvusToolsOpen);
  }
}

function toggleMilvusTools(){
  if(!milvusDiagnosticsEnabled||backend!=="milvus")return;
  milvusToolsOpen=!milvusToolsOpen;
  updateMilvusToolsVisibility();
  if(milvusToolsOpen)milvusDiagnosticStatus();
}

function setBackend(v){
  backend=v;
  ["pg","milvus","oracle","qdrant","elastic"].forEach(x=>{const el=document.getElementById(x+"Tab");if(el)el.classList.remove("active")});
  document.getElementById((v==="elasticsearch"?"elastic":v)+"Tab").classList.add("active");
  document.getElementById("backendText").innerText="Active backend: "+names[v];
  document.getElementById("uploadStatus").innerText="";
  document.getElementById("docsBox").innerHTML="";
  milvusToolsOpen=false;
  updateMilvusToolsVisibility();
}
function baseUrl(){return backend==="elasticsearch"?"/elasticsearch":"/"+backend}
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function addMessage(text,type){const box=document.getElementById("chatBox");const d=document.createElement("div");d.className="msg "+type;d.innerText=text;box.appendChild(d);box.scrollTop=box.scrollHeight;return d}
function formatAnswer(text){return esc(text||"No answer.").split(/
?
/).map(x=>x.trim()?`<p>${x.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>")}</p>`:"").join("")}
function renderBot(el,data){el.className="msg bot formatted";let h=`<div class="label">Answer</div><div>${formatAnswer(data.answer)}</div>`;if(data.sources?.length){h+=`<div class="section"><div class="label">Sources</div>`;data.sources.forEach((s,i)=>{const r=Number(s.rerank_score);const rs=Number.isFinite(r)?` | rerank ${r.toFixed(4)}`:"";h+=`<div class="source"><span class="rank">${i+1}</span><div><strong>${esc(s.file_name)}</strong><div class="muted">chunk ${esc(s.chunk_index)}</div></div><div class="muted">sim ${Number(s.similarity).toFixed(4)}${rs}</div></div>`});h+="</div>"}if(data.timings){h+=`<div class="section"><div class="label">Timings</div><div class="chips">`;Object.entries(data.timings).forEach(([k,v])=>h+=`<span class="chip">${esc(k)}: ${esc(v)}s</span>`);h+="</div></div>"}el.innerHTML=h}
function timingText(t){return t?"
"+Object.entries(t).map(([k,v])=>`${k}: ${v}s`).join("
"):""}
async function pollJob(id,b){const status=document.getElementById("uploadStatus");while(activeUploadJobId===id){try{const r=await fetch(`/upload-jobs/${id}`);const d=await r.json();status.innerText=`Job: ${d.job_id}
Status: ${d.status}
${d.message||""}
Files: ${d.files_done||0}/${d.files_total||0}
Chunks: ${d.chunks_done||0}/${d.chunks_total||0}
Inserted: ${d.inserted_chunks||0}${d.error?"
Error: "+d.error:""}${d.status==="success"?timingText(d.timings):""}`;if(["success","failed"].includes(d.status)){activeUploadJobId=null;if(d.status==="success"&&backend===b)listDocs();return}}catch(e){status.innerText="Job poll error: "+e;return}await new Promise(x=>setTimeout(x,2000))}}
async function uploadPdfs(){const input=document.getElementById("pdfFiles"),status=document.getElementById("uploadStatus"),b=backend;if(!input.files.length){status.innerText="Select at least one PDF.";return}const fd=new FormData();for(const f of input.files)fd.append("files",f);status.innerText="Queueing upload...";try{const r=await fetch(baseUrl()+"/upload-pdfs",{method:"POST",body:fd});const d=await r.json();if(!r.ok){status.innerText="Error: "+JSON.stringify(d);return}activeUploadJobId=d.job_id;pollJob(d.job_id,b)}catch(e){status.innerText="Request error: "+e}}
async function listDocs(){const box=document.getElementById("docsBox");box.innerHTML="Loading...";try{const r=await fetch(baseUrl()+"/docs");const d=await r.json();if(!r.ok){box.innerText=JSON.stringify(d);return}if(!d.documents?.length){box.innerHTML="<p>No documents.</p>";return}let h="<table><tr><th>File</th><th>Chunks</th></tr>";d.documents.forEach(x=>h+=`<tr><td>${esc(x.file_name)}</td><td>${esc(x.chunks)}</td></tr>`);box.innerHTML=h+"</table>"}catch(e){box.innerText="Error: "+e}}
async function clearDocs(){const s=document.getElementById("uploadStatus");if(activeUploadJobId){s.innerText="Wait for the upload job to finish.";return}try{const r=await fetch(baseUrl()+"/clear",{method:"DELETE"});const d=await r.json();s.innerText=d.message||JSON.stringify(d);listDocs()}catch(e){s.innerText="Error: "+e}}
async function askQuestion(){const q=document.getElementById("question"),text=q.value.trim();if(!text)return;addMessage(text,"user");q.value="";const loading=addMessage("Searching documents...","bot");try{const r=await fetch(baseUrl()+"/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:text})});const d=await r.json();if(!r.ok){loading.innerText="Error: "+JSON.stringify(d);return}renderBot(loading,d)}catch(e){loading.innerText="Request error: "+e}}

async function milvusDiagnosticStatus(){
  const output=document.getElementById("milvusDiagnosticsOutput");
  output.innerText="Checking Milvus...";
  try{
    const response=await fetch("/milvus/diagnostics");
    const data=await response.json();
    output.innerText=JSON.stringify(data,null,2);
  }catch(error){
    output.innerText="Diagnostic request error: "+error;
  }
}

async function milvusDiagnosticAction(action){
  const output=document.getElementById("milvusDiagnosticsOutput");
  output.innerText=`Running Milvus action: ${action}...`;
  try{
    const response=await fetch(`/milvus/diagnostics/${action}`,{method:"POST"});
    const data=await response.json();
    output.innerText=JSON.stringify(data,null,2);
  }catch(error){
    output.innerText="Milvus action error: "+error;
  }
}

document.getElementById("question").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();askQuestion()}});
loadAppConfig();
</script>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================


@app.get("/upload-jobs/{job_id}")
def upload_job_status(job_id: str):
    job = get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return job


@app.get("/app-config")
def app_config():
    return {
        "milvus_diagnostics_enabled": MILVUS_DIAGNOSTICS_ENABLED,
        "milvus_detailed_progress": MILVUS_DETAILED_PROGRESS,
    }


def require_milvus_diagnostics() -> None:
    if not MILVUS_DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=404, detail="Milvus diagnostics are disabled.")


@app.get("/milvus/diagnostics")
def milvus_diagnostics():
    require_milvus_diagnostics()
    started = now()
    try:
        connect_milvus()
        collections = utility.list_collections()
        exists = MILVUS_COLLECTION in collections
        result: Dict[str, Any] = {
            "status": "ok",
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "collection": MILVUS_COLLECTION,
            "collection_exists": exists,
            "collections": collections,
            "index_type_config": MILVUS_INDEX_TYPE,
            "consistency_level": MILVUS_CONSISTENCY_LEVEL,
            "operation_timeout": MILVUS_OPERATION_TIMEOUT,
            "index_timeout": MILVUS_INDEX_TIMEOUT,
            "flush_after_upload": MILVUS_FLUSH_AFTER_UPLOAD,
            "flush_timeout": MILVUS_FLUSH_TIMEOUT,
            "loaded_cache_flag": MILVUS_LOADED,
        }
        if exists:
            collection = Collection(MILVUS_COLLECTION)
            result["schema"] = [
                {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "is_primary": bool(getattr(field, "is_primary", False)),
                }
                for field in collection.schema.fields
            ]
            result["indexes"] = [
                {
                    "field_name": getattr(index, "field_name", None),
                    "params": getattr(index, "params", None),
                }
                for index in collection.indexes
            ]
            if hasattr(utility, "load_state"):
                try:
                    result["load_state"] = str(utility.load_state(MILVUS_COLLECTION))
                except Exception as exc:
                    result["load_state_error"] = str(exc)
        result["duration"] = elapsed(started)
        return result
    except Exception as exc:
        logger.exception("Milvus diagnostics failed")
        raise HTTPException(status_code=503, detail=f"Milvus diagnostics failed: {exc}") from exc


@app.post("/milvus/diagnostics/{action}")
def milvus_diagnostics_action(action: str):
    global MILVUS_LOADED
    require_milvus_diagnostics()
    allowed = {"reconnect", "load", "release", "flush", "create_index"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    started = now()
    try:
        if action == "reconnect":
            try:
                connections.disconnect(alias="default")
            except Exception:
                pass
            connect_milvus()
            MILVUS_LOADED = False
        else:
            collection = get_milvus_collection()
            if action == "load":
                collection.load(timeout=MILVUS_OPERATION_TIMEOUT)
                MILVUS_LOADED = True
            elif action == "release":
                collection.release()
                MILVUS_LOADED = False
            elif action == "flush":
                collection.flush(timeout=MILVUS_FLUSH_TIMEOUT)
            elif action == "create_index":
                ensure_milvus_index(collection)

        return {
            "status": "success",
            "action": action,
            "duration": elapsed(started),
            "loaded_cache_flag": MILVUS_LOADED,
        }
    except Exception as exc:
        logger.exception("Milvus diagnostic action failed action=%s", action)
        raise HTTPException(
            status_code=500,
            detail=f"Milvus action {action} failed: {exc}",
        ) from exc


@app.post("/pg/upload-pdfs", status_code=202)
async def pg_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("pg", background_tasks, pg_process_upload_files, files)


@app.post("/milvus/upload-pdfs", status_code=202)
async def milvus_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("milvus", background_tasks, milvus_process_upload_files, files)


@app.post("/oracle/upload-pdfs", status_code=202)
async def oracle_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("oracle", background_tasks, oracle_process_upload_files, files)


@app.post("/qdrant/upload-pdfs", status_code=202)
async def qdrant_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("qdrant", background_tasks, qdrant_process_upload_files, files)


@app.post("/elasticsearch/upload-pdfs", status_code=202)
async def elastic_upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job(
        "elasticsearch", background_tasks, elasticsearch_process_upload_files, files
    )


@app.post("/pg/ask")
def pg_ask(req: AskRequest):
    return run_ask_pipeline(req.question, pg_retrieve)


@app.post("/milvus/ask")
def milvus_ask(req: AskRequest):
    return run_ask_pipeline(req.question, milvus_retrieve)


@app.post("/oracle/ask")
def oracle_ask(req: AskRequest):
    return run_ask_pipeline(req.question, oracle_retrieve)


@app.post("/qdrant/ask")
def qdrant_ask(req: AskRequest):
    return run_ask_pipeline(req.question, qdrant_retrieve)


@app.post("/elasticsearch/ask")
def elasticsearch_ask(req: AskRequest):
    return run_ask_pipeline(req.question, elasticsearch_retrieve)


@app.get("/pg/docs")
def pg_docs():
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT file_name, COUNT(*) FROM {PG_TABLE} GROUP BY file_name ORDER BY file_name"
            )
            rows = cur.fetchall()
        return {"documents": [{"file_name": x[0], "chunks": int(x[1])} for x in rows]}
    finally:
        conn.close()


@app.get("/oracle/docs")
def oracle_docs():
    conn = get_oracle_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT file_name, COUNT(*) FROM {ORACLE_TABLE} GROUP BY file_name ORDER BY file_name"
        )
        rows = cur.fetchall()
        return {"documents": [{"file_name": x[0], "chunks": int(x[1])} for x in rows]}
    finally:
        conn.close()


@app.get("/milvus/docs")
def milvus_docs():
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        return {"documents": []}
    collection = Collection(MILVUS_COLLECTION)
    if not milvus_collection_has_index(collection):
        return {
            "documents": [],
            "warning": "Milvus collection exists but its vector index has not been created yet.",
        }
    ensure_milvus_loaded(collection)
    rows = milvus_query_with_consistency(
        collection,
        expr="id >= 0",
        output_fields=["file_name"],
        limit=10000,
        timeout=MILVUS_OPERATION_TIMEOUT,
    )
    counts: Dict[str, int] = {}
    for row in rows:
        name = row.get("file_name")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return {"documents": [{"file_name": k, "chunks": v} for k, v in sorted(counts.items())]}


@app.get("/qdrant/docs")
def qdrant_docs():
    client = ensure_qdrant_collection()
    offset = None
    counts: Dict[str, int] = {}
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            name = (point.payload or {}).get("file_name")
            if name:
                counts[name] = counts.get(name, 0) + 1
        if offset is None:
            break
    return {"documents": [{"file_name": k, "chunks": v} for k, v in sorted(counts.items())]}


@app.get("/elasticsearch/docs")
def elasticsearch_docs():
    client = get_es_client()
    if not bool(client.indices.exists(index=ES_INDEX)):
        return {"documents": []}
    response = client.search(
        index=ES_INDEX,
        size=0,
        aggs={"documents": {"terms": {"field": "file_name", "size": 10000}}},
    )
    buckets = response.get("aggregations", {}).get("documents", {}).get("buckets", [])
    return {
        "documents": [
            {"file_name": x.get("key"), "chunks": int(x.get("doc_count", 0))}
            for x in buckets
        ]
    }


@app.delete("/pg/clear")
def pg_clear():
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {PG_TABLE} RESTART IDENTITY")
        return {"status": "success", "message": "PGVector documents cleared."}
    finally:
        conn.close()


@app.delete("/oracle/clear")
def oracle_clear():
    conn = get_oracle_connection()
    try:
        cur = conn.cursor()
        cur.execute(f"TRUNCATE TABLE {ORACLE_TABLE}")
        return {"status": "success", "message": "Oracle documents cleared."}
    finally:
        conn.close()


@app.delete("/milvus/clear")
def milvus_clear():
    global MILVUS_LOADED
    connect_milvus()
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
    MILVUS_LOADED = False
    return {"status": "success", "message": "Milvus collection cleared."}


@app.delete("/qdrant/clear")
def qdrant_clear():
    client = get_qdrant_client()
    names = [x.name for x in client.get_collections().collections]
    if QDRANT_COLLECTION in names:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
    return {"status": "success", "message": "Qdrant collection cleared."}


@app.delete("/elasticsearch/clear")
def elasticsearch_clear():
    client = get_es_client()
    if bool(client.indices.exists(index=ES_INDEX)):
        client.indices.delete(index=ES_INDEX)
    return {"status": "success", "message": "Elasticsearch index cleared."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": {"host": APP_HOST, "port": APP_PORT, "workers": 1},
        "embedding": {"model": REMOTE_MODEL_NAME, "dim": EMBEDDING_DIM},
        "retrieval": {
            "top_k": TOP_K,
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "threshold": SIMILARITY_THRESHOLD,
            "query_rewrite": ENABLE_QUERY_REWRITE,
        },
        "rerank": {
            "mode": RERANK_MODE,
            "style": RERANK_API_STYLE,
            "model": RERANK_MODEL,
            "max_candidates": RERANK_MAX_CANDIDATES,
        },
        "pg": {"table": PG_TABLE, "candidate_k": PG_CANDIDATE_K},
        "milvus": {
            "collection": MILVUS_COLLECTION,
            "index_type": MILVUS_INDEX_TYPE,
            "consistency_level": MILVUS_CONSISTENCY_LEVEL,
            "operation_timeout": MILVUS_OPERATION_TIMEOUT,
            "flush_after_upload": MILVUS_FLUSH_AFTER_UPLOAD,
            "flush_timeout": MILVUS_FLUSH_TIMEOUT,
            "detailed_progress": MILVUS_DETAILED_PROGRESS,
            "diagnostics_enabled": MILVUS_DIAGNOSTICS_ENABLED,
        },
        "oracle": {
            "table": ORACLE_TABLE,
            "approx": ORACLE_USE_APPROX_SEARCH,
            "target_accuracy": ORACLE_TARGET_ACCURACY,
        },
        "qdrant": {"collection": QDRANT_COLLECTION},
        "elasticsearch": {"index": ES_INDEX, "url": ES_URL},
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting on %s:%s with workers=1", APP_HOST, APP_PORT)
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        workers=1,
        log_level=APP_LOG_LEVEL,
    )
