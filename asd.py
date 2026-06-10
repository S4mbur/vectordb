# -*- coding: utf-8 -*-

import os
import uuid
import time
import logging
import threading
import json
import re
from array import array
from typing import List, Dict, Any

import requests
import psycopg2
from psycopg2.extras import execute_values

import oracledb

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from elasticsearch import Elasticsearch, helpers

load_dotenv()

# =========================
# CONFIG
# =========================

PG_HOST = os.getenv("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DBNAME = os.getenv("PG_DBNAME")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE = os.getenv("PG_TABLE", "ai_docs")
PG_CANDIDATE_K = int(os.getenv("PG_CANDIDATE_K", "30"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ai_docs_milvus")
MILVUS_INSERT_BATCH_SIZE = int(os.getenv("MILVUS_INSERT_BATCH_SIZE", "64"))
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "FLAT").upper()

ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_DSN = os.getenv("ORACLE_DSN")
ORACLE_TABLE = os.getenv("ORACLE_TABLE", "AI_DOCS_ORACLE")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_docs_qdrant")
QDRANT_INSERT_BATCH_SIZE = int(os.getenv("QDRANT_INSERT_BATCH_SIZE", "64"))

ES_URL = os.getenv("ES_URL", os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_CA_CERTS = os.getenv("ES_CA_CERTS")
ES_DEFAULT_VERIFY_CERTS = "false" if ES_URL.lower().startswith("http://") else "true"
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", ES_DEFAULT_VERIFY_CERTS).lower() == "true"
ES_INDEX = os.getenv("ES_INDEX", "ai_docs_elasticsearch")
ES_INSERT_BATCH_SIZE = int(os.getenv("ES_INSERT_BATCH_SIZE", "64"))
ES_NUM_CANDIDATES = int(os.getenv("ES_NUM_CANDIDATES", "100"))
ES_REQUEST_TIMEOUT = int(os.getenv("ES_REQUEST_TIMEOUT", "120"))

REMOTE_API_URL = os.getenv("REMOTE_API_URL")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY")
REMOTE_MODEL_NAME = os.getenv("REMOTE_MODEL_NAME")

REMOTE_CHAT_URL = os.getenv("REMOTE_CHAT_URL")
REMOTE_CHAT_MODEL = os.getenv("REMOTE_CHAT_MODEL")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
HALFVEC_DIM = int(os.getenv("HALFVEC_DIM", "2048"))
TOP_K = int(os.getenv("TOP_K", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "18000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "450"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "80"))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "20"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
UPLOAD_JOB_HISTORY_LIMIT = int(os.getenv("UPLOAD_JOB_HISTORY_LIMIT", "100"))
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "30"))
ENABLE_LLM_RERANK = os.getenv("ENABLE_LLM_RERANK", "false").lower() == "true"
RERANK_MAX_CANDIDATES = int(os.getenv("RERANK_MAX_CANDIDATES", str(RETRIEVAL_CANDIDATE_K)))
RERANK_CHUNK_MAX_CHARS = int(os.getenv("RERANK_CHUNK_MAX_CHARS", "1200"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")

NOT_FOUND_TR = "Bu bilgi yuklenen dokumanlarda bulunamadi."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-vector-rag-app")

app = FastAPI(title="PGVector + Milvus + Oracle 26ai + Qdrant + Elasticsearch PDF RAG")


class AskRequest(BaseModel):
    question: str


UPLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
UPLOAD_JOB_LOCK = threading.Lock()


# =========================
# COMMON UTILS
# =========================

def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def vector_to_sql(v: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def vector_to_halfvec_sql(v: List[float]) -> str:
    half = v[:HALFVEC_DIM]
    return "[" + ",".join(str(float(x)) for x in half) + "]"


def to_oracle_float32_vector(v: List[float]):
    return array("f", [float(x) for x in v])


def chunk_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def parse_embedding_response(data: Dict[str, Any], expected_count: int) -> List[List[float]]:
    try:
        items = data["data"]
    except Exception:
        logger.error("Embedding response: %s", data)
        raise HTTPException(status_code=500, detail="Bad embedding response.")

    if all(isinstance(item, dict) and "index" in item for item in items):
        items = sorted(items, key=lambda item: item["index"])

    embeddings = []

    for item in items:
        try:
            emb = item["embedding"]
        except Exception:
            logger.error("Embedding response: %s", data)
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


def request_embeddings(inputs) -> List[List[float]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }

    expected_count = len(inputs) if isinstance(inputs, list) else 1

    payload = {
        "model": REMOTE_MODEL_NAME,
        "input": inputs,
    }

    resp = requests.post(
        REMOTE_API_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if resp.status_code != 200:
        logger.error("Embedding API error: %s - %s", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Embedding API error.")

    return parse_embedding_response(resp.json(), expected_count)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    embeddings = []

    for batch in chunk_batches(texts, batch_size):
        if len(batch) == 1:
            embeddings.extend(request_embeddings(batch[0]))
        else:
            embeddings.extend(request_embeddings(batch))

    return embeddings


def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]


def call_chat_model(system_prompt: str, user_prompt: str, timeout: int = 180) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }

    payload = {
        "model": REMOTE_CHAT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(
        REMOTE_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        logger.error("Chat API error: %s - %s", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Chat API error.")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def ask_llm(question: str, context: str) -> str:
    system_prompt = """
You are a strict document-based RAG assistant.
Use only the provided context.
If the context contains enough information to answer, answer clearly in Turkish.
Keep the answer readable: use short paragraphs, bullet lists for enumerations, and avoid long unbroken blocks.
Do not include a separate sources section; the application will render sources separately.
If the answer is not supported by the context, say exactly:
Bu bilgi yuklenen dokumanlarda bulunamadi.

Do not use outside knowledge.
Do not guess.
"""

    return call_chat_model(
        system_prompt,
        f"Context:\n{context}\n\nQuestion:\n{question}",
        timeout=180,
    )


def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n--- PAGE {page_no} ---\n{text}")

    return "\n".join(pages)


def normalize_text_space(text: str) -> str:
    return " ".join((text or "").split())


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def split_text_pages(text: str):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    marker_re = re.compile(r"---\s*PAGE\s+(\d+)\s*---", re.IGNORECASE)
    matches = list(marker_re.finditer(normalized))

    if not matches:
        return [(None, normalized)]

    pages = []

    for idx, match in enumerate(matches):
        page_no = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        pages.append((page_no, normalized[start:end].strip()))

    return pages


def split_paragraphs(text: str) -> List[str]:
    paragraphs = []

    for paragraph in re.split(r"\n\s*\n+", text or ""):
        normalized = normalize_text_space(paragraph)

        if normalized:
            paragraphs.append(normalized)

    return paragraphs


def split_sentences(text: str) -> List[str]:
    normalized = normalize_text_space(text)

    if not normalized:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]

    return sentences or [normalized]


def split_words_window(text: str, max_words: int, overlap_words: int = 0) -> List[str]:
    words = normalize_text_space(text).split()

    if not words:
        return []

    step = max(1, max_words - max(0, min(overlap_words, max_words - 1)))

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), step)
    ]


def split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    paragraph = normalize_text_space(paragraph)

    if count_words(paragraph) <= max_words:
        return [paragraph] if paragraph else []

    units = []
    current = []
    current_words = 0

    for sentence in split_sentences(paragraph):
        sentence_words = count_words(sentence)

        if sentence_words > max_words:
            if current:
                units.append(" ".join(current).strip())
                current = []
                current_words = 0

            units.extend(split_words_window(sentence, max_words, CHUNK_OVERLAP_WORDS))
            continue

        if current and current_words + sentence_words > max_words:
            units.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        units.append(" ".join(current).strip())

    return [unit for unit in units if unit]


def build_page_chunks(page_no, units: List[str], max_words: int, overlap_words: int) -> List[str]:
    chunks = []
    current_parts = []
    current_words = 0
    current_has_new_content = False
    overlap_words = max(0, min(overlap_words, max_words - 1))

    def flush_current(carry_overlap: bool):
        nonlocal current_parts, current_words, current_has_new_content

        if current_has_new_content:
            chunk = normalize_text_space(" ".join(current_parts))

            if chunk:
                if page_no:
                    chunk = f"--- PAGE {page_no} ---\n{chunk}"

                chunks.append(chunk)

            if carry_overlap and overlap_words > 0:
                plain_chunk = normalize_text_space(" ".join(current_parts))
                overlap_text = " ".join(plain_chunk.split()[-overlap_words:])
                current_parts = [overlap_text] if overlap_text else []
                current_words = count_words(overlap_text)
                current_has_new_content = False
                return

        current_parts = []
        current_words = 0
        current_has_new_content = False

    for unit in units:
        unit = normalize_text_space(unit)
        unit_words = count_words(unit)

        if not unit or unit_words == 0:
            continue

        if current_has_new_content and current_words + unit_words > max_words:
            flush_current(carry_overlap=True)

        if not current_has_new_content and current_words + unit_words > max_words:
            current_parts = []
            current_words = 0

        current_parts.append(unit)
        current_words += unit_words
        current_has_new_content = True

    flush_current(carry_overlap=False)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_MAX_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    max_words = max(1, chunk_size)
    overlap_words = max(0, overlap)
    raw_chunks = []

    for page_no, page_text in split_text_pages(text):
        units = []

        for paragraph in split_paragraphs(page_text):
            units.extend(split_long_paragraph(paragraph, max_words))

        raw_chunks.extend(build_page_chunks(page_no, units, max_words, overlap_words))

    min_words = max(1, CHUNK_MIN_WORDS)
    chunks = [chunk for chunk in raw_chunks if count_words(chunk) >= min_words]

    return chunks


def build_context_from_items(items: List[Dict[str, Any]]):
    context = ""
    sources = []

    for item in items:
        piece = (
            f"\nSOURCE: {item['file_name']} | CHUNK: {item['chunk_index']} "
            f"| SIMILARITY: {item['similarity']:.4f}\n"
            f"{item['content']}\n"
        )

        if len(context) + len(piece) > MAX_CONTEXT_LENGTH:
            break

        context += piece
        sources.append(
            {
                "file_name": item["file_name"],
                "chunk_index": item["chunk_index"],
                "similarity": item["similarity"],
            }
        )

    return context, sources


def empty_result(total_start, embed_time=0, retrieval_time=0, rewrite_time=0, rerank_time=0):
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


def retrieval_candidate_limit() -> int:
    return max(TOP_K, RETRIEVAL_CANDIDATE_K)


def rewrite_question_for_search(question: str) -> str:
    system_prompt = """
You rewrite user questions into concise vector-search queries.
Expand abbreviations, product names, error codes, and likely technical terms when useful.
Return only the rewritten search query. Do not answer the question.
"""
    user_prompt = f"""
Original question:
{question}

Rewrite it for semantic/vector retrieval. Keep it under 40 words.
"""

    try:
        rewritten = call_chat_model(system_prompt, user_prompt, timeout=60)
        rewritten = normalize_text_space(rewritten.strip().strip('"').strip("'"))

        if not rewritten:
            return question

        return rewritten
    except Exception as exc:
        logger.warning("Question rewrite failed, falling back to original question: %s", exc)
        return question


def prepare_query_embedding(question: str):
    s = now()
    search_query = rewrite_question_for_search(question)
    rewrite_time = elapsed(s)

    s = now()
    q_emb = get_embedding(search_query)
    embed_time = elapsed(s)

    return search_query, q_emb, rewrite_time, embed_time


def sort_chunks_by_similarity(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: float(item.get("similarity") or 0),
        reverse=True,
    )[:TOP_K]


def extract_json_payload(text: str):
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


def parse_rerank_scores(response_text: str) -> Dict[int, float]:
    payload = extract_json_payload(response_text)

    if isinstance(payload, dict):
        payload = payload.get("scores") or payload.get("results") or payload.get("rankings")

    if not isinstance(payload, list):
        raise ValueError("Rerank response must be a JSON list.")

    scores = {}

    for item in payload:
        if not isinstance(item, dict):
            continue

        raw_id = item.get("id", item.get("index"))
        raw_score = item.get("score", item.get("relevance"))

        if raw_id is None or raw_score is None:
            continue

        idx = int(raw_id)
        score = float(raw_score)
        scores[idx] = max(0.0, min(1.0, score))

    if not scores:
        raise ValueError("Rerank response did not contain scores.")

    return scores


def rerank_chunks(question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    fallback = sort_chunks_by_similarity(candidates)

    if not ENABLE_LLM_RERANK:
        return fallback

    limited_candidates = candidates[:max(TOP_K, RERANK_MAX_CANDIDATES)]
    candidate_lines = []

    for idx, item in enumerate(limited_candidates):
        content = normalize_text_space(str(item.get("content") or ""))[:RERANK_CHUNK_MAX_CHARS]
        candidate_lines.append(
            "\n".join(
                [
                    f"ID: {idx}",
                    f"File: {item.get('file_name')}",
                    f"Chunk: {item.get('chunk_index')}",
                    f"Vector similarity: {float(item.get('similarity') or 0):.4f}",
                    f"Text: {content}",
                ]
            )
        )

    system_prompt = """
You are a strict reranker for document retrieval.
Score each candidate chunk for how useful it is to answer the user's original question.
Return only JSON. Use this exact shape:
[{"id": 0, "score": 0.0}]
Scores must be between 0 and 1.
"""
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Candidate chunks:\n"
        + "\n\n---\n\n".join(candidate_lines)
    )

    try:
        response_text = call_chat_model(system_prompt, user_prompt, timeout=90)
        score_by_id = parse_rerank_scores(response_text)
    except Exception as exc:
        logger.warning("LLM rerank failed, using vector scores: %s", exc)
        return fallback

    reranked = []

    for idx, item in enumerate(limited_candidates):
        copied = dict(item)
        copied["rerank_score"] = score_by_id.get(idx, float(item.get("similarity") or 0))
        reranked.append(copied)

    return sorted(
        reranked,
        key=lambda item: (
            float(item.get("rerank_score") or 0),
            float(item.get("similarity") or 0),
        ),
        reverse=True,
    )[:TOP_K]


def elastic_score_to_cosine(score: float) -> float:
    similarity = (float(score) * 2.0) - 1.0
    return max(-1.0, min(1.0, similarity))


def copy_upload_job(job: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(job)
    copied["files"] = list(job.get("files", []))
    copied["timings"] = dict(job.get("timings", {}))

    if job.get("result"):
        copied["result"] = dict(job["result"])

    return copied


def prune_upload_jobs_locked():
    if len(UPLOAD_JOBS) <= UPLOAD_JOB_HISTORY_LIMIT:
        return

    finished = [
        job
        for job in UPLOAD_JOBS.values()
        if job.get("status") in ("success", "failed")
    ]
    finished.sort(key=lambda job: job.get("updated_at", 0))

    for job in finished:
        if len(UPLOAD_JOBS) <= UPLOAD_JOB_HISTORY_LIMIT:
            break

        UPLOAD_JOBS.pop(job["job_id"], None)


def create_upload_job(backend: str, saved_files: List[Dict[str, str]]) -> str:
    job_id = str(uuid.uuid4())
    ts = time.time()

    with UPLOAD_JOB_LOCK:
        UPLOAD_JOBS[job_id] = {
            "job_id": job_id,
            "backend": backend,
            "status": "queued",
            "message": "Queued.",
            "current_file": None,
            "created_at": ts,
            "updated_at": ts,
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


def get_upload_job(job_id: str):
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return None

        return copy_upload_job(job)


def update_upload_job(job_id: str, **updates):
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return

        job.update(updates)
        job["updated_at"] = time.time()


def increment_upload_job(job_id: str, message: str = None, **increments):
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return

        for key, amount in increments.items():
            job[key] = job.get(key, 0) + amount

        if message is not None:
            job["message"] = message

        job["updated_at"] = time.time()


def record_upload_file_result(job_id: str, file_name: str, status: str, chunks: int):
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return

        for item in job.get("files", []):
            if item.get("file") == file_name and item.get("status") != "ok":
                item["status"] = status
                item["chunks"] = chunks
                break

        job["updated_at"] = time.time()


def finish_upload_job(job_id: str, result: Dict[str, Any]):
    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return

        job["status"] = "success"
        job["message"] = "Completed."
        job["inserted_chunks"] = result.get("inserted_chunks", job.get("inserted_chunks", 0))
        job["files"] = result.get("files", job.get("files", []))
        job["timings"] = result.get("timings", {})
        job["result"] = result
        job["updated_at"] = time.time()


def fail_upload_job(job_id: str, exc: Exception):
    detail = getattr(exc, "detail", None) or str(exc)

    with UPLOAD_JOB_LOCK:
        job = UPLOAD_JOBS.get(job_id)

        if not job:
            return

        job["status"] = "failed"
        job["message"] = "Failed."
        job["error"] = detail
        job["updated_at"] = time.time()


async def save_uploaded_pdfs(files: List[UploadFile]) -> List[Dict[str, str]]:
    os.makedirs("uploads", exist_ok=True)
    saved_files = []

    for uploaded_file in files:
        original_name = os.path.basename(uploaded_file.filename or "")

        if not original_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        safe_name = f"{uuid.uuid4()}_{original_name}"
        file_path = os.path.join("uploads", safe_name)

        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())

        saved_files.append({"original_name": original_name, "file_path": file_path})

    return saved_files


def embed_chunks_for_upload(job_id: str, file_name: str, chunks: List[str]):
    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    embeddings = []
    embed_time = 0

    for batch in chunk_batches(chunks, batch_size):
        s = now()
        inputs = batch[0] if len(batch) == 1 else batch
        batch_embeddings = request_embeddings(inputs)
        embed_time += elapsed(s)
        embeddings.extend(batch_embeddings)

        increment_upload_job(
            job_id,
            chunks_done=len(batch),
            message=f"Embedding {file_name}: {len(embeddings)}/{len(chunks)} chunks.",
        )

    return embeddings, embed_time


def run_upload_job(job_id: str, backend: str, saved_files: List[Dict[str, str]], process_func):
    update_upload_job(job_id, status="running", message=f"{backend} upload started.")

    try:
        result = process_func(job_id, saved_files)
        finish_upload_job(job_id, result)
    except Exception as exc:
        logger.exception("Upload job failed. job_id=%s backend=%s", job_id, backend)
        fail_upload_job(job_id, exc)


async def enqueue_upload_job(
    backend: str,
    background_tasks: BackgroundTasks,
    process_func,
    files: List[UploadFile],
):
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


# =========================
# CONNECTIONS
# =========================

def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=30,
    )


def get_milvus_collection():
    connect_milvus()

    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="PDF RAG collection",
            enable_dynamic_field=False,
        )

        collection = Collection(
            name=MILVUS_COLLECTION,
            schema=schema,
            using="default",
            shards_num=1,
        )

        if MILVUS_INDEX_TYPE == "HNSW":
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 64,
                },
            }
        elif MILVUS_INDEX_TYPE == "IVF_FLAT":
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {
                    "nlist": 128,
                },
            }
        else:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT",
                "params": {},
            }

        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

    else:
        collection = Collection(MILVUS_COLLECTION)

    return collection


def get_oracle_connection():
    return oracledb.connect(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=ORACLE_DSN,
    )


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_qdrant_collection():
    client = get_qdrant_client()

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    return client


def get_es_client():
    hosts = [x.strip() for x in ES_URL.split(",") if x.strip()]
    use_https = any(host.lower().startswith("https://") for host in hosts)

    kwargs = {
        "request_timeout": ES_REQUEST_TIMEOUT,
    }

    if use_https:
        kwargs["verify_certs"] = ES_VERIFY_CERTS

    if use_https and ES_CA_CERTS:
        kwargs["ca_certs"] = ES_CA_CERTS

    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USERNAME and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)

    return Elasticsearch(hosts, **kwargs)


def ensure_es_index():
    if EMBEDDING_DIM > 4096:
        raise HTTPException(
            status_code=500,
            detail="Elasticsearch dense_vector supports EMBEDDING_DIM <= 4096.",
        )

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
        return client

    mapping = client.indices.get_mapping(index=ES_INDEX)
    embedding_field = (
        mapping.get(ES_INDEX, {})
        .get("mappings", {})
        .get("properties", {})
        .get("embedding", {})
    )

    if embedding_field and int(embedding_field.get("dims", EMBEDDING_DIM)) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Elasticsearch index dim mismatch. "
                f"Expected={EMBEDDING_DIM}, Got={embedding_field.get('dims')}"
            ),
        )

    return client


# =========================
# UI
# =========================

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Multi Vector RAG Chat</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f4f4f5; color: #111827; }
    .container { max-width: 1080px; margin: 35px auto; padding: 20px; }
    .card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom: 20px; }
    .tabs { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
    .tab { padding: 12px 18px; border-radius: 10px; border: 1px solid #d1d5db; background: #f9fafb; cursor: pointer; font-weight: 700; }
    .tab.active { background: #111827; color: white; }
    .upload-area { border: 2px dashed #9ca3af; border-radius: 14px; padding: 24px; text-align: center; background: #fafafa; }
    button { border: none; border-radius: 10px; padding: 12px 18px; background: #111827; color: white; cursor: pointer; font-weight: 600; margin: 8px 4px 0 4px; }
    textarea { width: 100%; min-height: 90px; resize: vertical; border-radius: 12px; border: 1px solid #d1d5db; padding: 14px; font-size: 15px; box-sizing: border-box; }
    .chat-box { min-height: 300px; max-height: 540px; overflow-y: auto; background: #f9fafb; border-radius: 14px; padding: 18px; border: 1px solid #e5e7eb; }
    .message { padding: 14px; border-radius: 14px; margin-bottom: 12px; line-height: 1.5; white-space: pre-wrap; }
    .user { background: #e0f2fe; margin-left: 80px; }
    .bot { background: white; border: 1px solid #e5e7eb; margin-right: 80px; }
    .bot.formatted { white-space: normal; }
    .answer-title { color: #6b7280; font-size: 12px; font-weight: 700; letter-spacing: 0; text-transform: uppercase; margin-bottom: 6px; }
    .answer-text p { margin: 0 0 10px 0; }
    .answer-text p:last-child { margin-bottom: 0; }
    .answer-text ul, .answer-text ol { margin: 6px 0 10px 22px; padding: 0; }
    .answer-text li { margin: 4px 0; }
    .meta-section { margin-top: 14px; padding-top: 12px; border-top: 1px solid #e5e7eb; }
    .source-list { display: grid; gap: 8px; margin-top: 8px; }
    .source-item { display: grid; grid-template-columns: 28px 1fr auto; gap: 8px; align-items: center; padding: 8px; border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb; }
    .source-rank { width: 24px; height: 24px; border-radius: 50%; background: #111827; color: white; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; }
    .source-file { font-weight: 700; word-break: break-word; }
    .source-meta { color: #6b7280; font-size: 12px; white-space: nowrap; }
    .timing-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .timing-chip { border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb; padding: 6px 8px; font-size: 12px; color: #374151; }
    .status { font-size: 14px; color: #6b7280; margin-top: 10px; white-space: pre-wrap; }
    .row { display: flex; gap: 10px; align-items: flex-start; }
    .row textarea { flex: 1; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
    th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 9px; }
    th { background: #f9fafb; }
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

      <div class="upload-area">
        <strong id="backendText">Active backend: PGVector</strong><br />
        <input id="pdfFiles" type="file" accept="application/pdf" multiple />
        <br />
        <button onclick="uploadPdfs()">Upload PDFs</button>
        <button onclick="listDocs()">List Uploaded PDFs</button>
        <button onclick="clearDocs()">Clear Active DB</button>
        <div id="uploadStatus" class="status"></div>
      </div>

      <div id="docsBox"></div>
    </div>

    <div class="card">
      <div id="chatBox" class="chat-box">
        <div class="message bot">Upload PDFs, then ask a question.</div>
      </div>

      <br />

      <div class="row">
        <textarea id="question" placeholder="Ask a question based on uploaded documents..."></textarea>
        <button onclick="askQuestion()">Ask</button>
      </div>
    </div>
  </div>

<script>
let backend = "pg";
let activeUploadJobId = null;

function setBackend(value) {
  backend = value;

  document.getElementById("pgTab").classList.remove("active");
  document.getElementById("milvusTab").classList.remove("active");
  document.getElementById("oracleTab").classList.remove("active");
  document.getElementById("qdrantTab").classList.remove("active");
  document.getElementById("elasticTab").classList.remove("active");

  if (value === "pg") {
    document.getElementById("pgTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: PGVector";
  } else if (value === "milvus") {
    document.getElementById("milvusTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Milvus";
  } else if (value === "oracle") {
    document.getElementById("oracleTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Oracle 26ai";
  } else if (value === "qdrant") {
    document.getElementById("qdrantTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Qdrant";
  } else {
    document.getElementById("elasticTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Elasticsearch";
  }

  document.getElementById("uploadStatus").innerText = "";
  document.getElementById("docsBox").innerHTML = "";
}

function baseUrl() {
  if (backend === "pg") return "/pg";
  if (backend === "milvus") return "/milvus";
  if (backend === "oracle") return "/oracle";
  if (backend === "qdrant") return "/qdrant";
  return "/elasticsearch";
}

function addMessage(text, type) {
  const chatBox = document.getElementById("chatBox");
  const div = document.createElement("div");
  div.className = "message " + type;
  div.innerText = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

function timingText(timings) {
  if (!timings) return "";
  return "\\n\\nTimings:\\n" + Object.entries(timings).map(([k,v]) => `${k}: ${v}s`).join("\\n");
}

function escapeHtml(value) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  };

  return String(value ?? "").replace(/[&<>"']/g, char => map[char]);
}

function formatInline(value) {
  return escapeHtml(value).replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");
}

function formatAnswerHtml(text) {
  const raw = String(text || "").trim();

  if (!raw) {
    return "<p>No answer.</p>";
  }

  const lines = raw.split(/\\r?\\n/);
  let html = "";
  let listType = null;

  function openList(type) {
    if (listType === type) return;
    closeList();
    listType = type;
    html += `<${type}>`;
  }

  function closeList() {
    if (!listType) return;
    html += `</${listType}>`;
    listType = null;
  }

  lines.forEach(line => {
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      return;
    }

    const bullet = trimmed.match(/^[-*]\\s+(.+)/);
    const numbered = trimmed.match(/^\\d+\\.\\s+(.+)/);

    if (bullet) {
      openList("ul");
      html += `<li>${formatInline(bullet[1])}</li>`;
      return;
    }

    if (numbered) {
      openList("ol");
      html += `<li>${formatInline(numbered[1])}</li>`;
      return;
    }

    closeList();
    html += `<p>${formatInline(trimmed)}</p>`;
  });

  closeList();
  return html;
}

function renderBotResponse(element, data) {
  element.className = "message bot formatted";

  let html = `
    <div>
      <div class="answer-title">Answer</div>
      <div class="answer-text">${formatAnswerHtml(data.answer || "No answer.")}</div>
    </div>
  `;

  if (data.sources && data.sources.length > 0) {
    html += `
      <div class="meta-section">
        <div class="answer-title">Sources</div>
        <div class="source-list">
    `;

    data.sources.forEach((source, index) => {
      const similarity = Number(source.similarity);
      const similarityText = Number.isFinite(similarity) ? similarity.toFixed(4) : "-";

      html += `
        <div class="source-item">
          <span class="source-rank">${index + 1}</span>
          <div>
            <div class="source-file">${escapeHtml(source.file_name)}</div>
            <div class="source-meta">chunk ${escapeHtml(source.chunk_index)}</div>
          </div>
          <div class="source-meta">${escapeHtml(similarityText)}</div>
        </div>
      `;
    });

    html += `
        </div>
      </div>
    `;
  }

  if (data.timings) {
    html += `
      <div class="meta-section">
        <div class="answer-title">Timings</div>
        <div class="timing-list">
    `;

    Object.entries(data.timings).forEach(([key, value]) => {
      html += `<span class="timing-chip">${escapeHtml(key)}: ${escapeHtml(value)}s</span>`;
    });

    html += `
        </div>
      </div>
    `;
  }

  element.innerHTML = html;
}

function uploadJobText(job) {
  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Message: ${job.message || ""}`,
    `Files: ${job.files_done || 0}/${job.files_total || 0}`,
    `Chunks embedded: ${job.chunks_done || 0}/${job.chunks_total || 0}`,
    `Chunks inserted: ${job.inserted_chunks || 0}`
  ];

  if (job.current_file) {
    lines.push(`Current file: ${job.current_file}`);
  }

  if (job.error) {
    lines.push(`Error: ${job.error}`);
  }

  if (job.status === "success" && job.timings) {
    lines.push(timingText(job.timings).trim());
  }

  return lines.join("\\n");
}

async function pollUploadJob(jobId, jobBackend) {
  const status = document.getElementById("uploadStatus");

  while (activeUploadJobId === jobId) {
    try {
      const res = await fetch(`/upload-jobs/${jobId}`);
      const data = await res.json();

      if (!res.ok) {
        status.innerText = "Job status error: " + JSON.stringify(data);
        return;
      }

      status.innerText = uploadJobText(data);

      if (data.status === "success") {
        activeUploadJobId = null;

        if (backend === jobBackend) {
          listDocs();
        }

        return;
      }

      if (data.status === "failed") {
        activeUploadJobId = null;
        return;
      }
    } catch (err) {
      status.innerText = "Job poll error: " + err;
      return;
    }

    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

async function uploadPdfs() {
  const input = document.getElementById("pdfFiles");
  const status = document.getElementById("uploadStatus");
  const uploadBackend = backend;

  if (!input.files.length) {
    status.innerText = "Select at least one PDF.";
    return;
  }

  const formData = new FormData();

  for (const file of input.files) {
    formData.append("files", file);
  }

  status.innerText = "Uploading files and queueing job...";

  try {
    const res = await fetch(baseUrl() + "/upload-pdfs", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      status.innerText = "Error: " + JSON.stringify(data);
      return;
    }

    activeUploadJobId = data.job_id;
    status.innerText = "Upload job queued. Job: " + data.job_id;
    pollUploadJob(data.job_id, uploadBackend);
  } catch (err) {
    status.innerText = "Request error: " + err;
  }
}

async function clearDocs() {
  const status = document.getElementById("uploadStatus");

  if (activeUploadJobId) {
    status.innerText = "An upload job is still running. Wait for it to finish before clearing.";
    return;
  }

  status.innerText = "Clearing...";

  try {
    const res = await fetch(baseUrl() + "/clear", { method: "DELETE" });
    const data = await res.json();

    if (!res.ok) {
      status.innerText = "Error: " + JSON.stringify(data);
      return;
    }

    status.innerText = data.message;
    listDocs();
  } catch (err) {
    status.innerText = "Request error: " + err;
  }
}

async function listDocs() {
  const box = document.getElementById("docsBox");
  box.innerHTML = "Loading documents...";

  try {
    const res = await fetch(baseUrl() + "/docs");
    const data = await res.json();

    if (!res.ok) {
      box.innerHTML = "Error: " + JSON.stringify(data);
      return;
    }

    if (!data.documents || data.documents.length === 0) {
      box.innerHTML = "<p>No uploaded PDFs found.</p>";
      return;
    }

    let html = "<table><thead><tr><th>File</th><th>Chunks</th></tr></thead><tbody>";

    data.documents.forEach(d => {
      html += `<tr><td>${escapeHtml(d.file_name)}</td><td>${escapeHtml(d.chunks)}</td></tr>`;
    });

    html += "</tbody></table>";
    box.innerHTML = html;
  } catch (err) {
    box.innerHTML = "Request error: " + err;
  }
}

async function askQuestion() {
  const q = document.getElementById("question");
  const question = q.value.trim();

  if (!question) return;

  addMessage(question, "user");
  q.value = "";

  const loadingMsg = addMessage("Searching documents...", "bot");

  try {
    const res = await fetch(baseUrl() + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await res.json();

    if (!res.ok) {
      loadingMsg.innerText = "Error: " + JSON.stringify(data);
      return;
    }

    renderBotResponse(loadingMsg, data);

  } catch (err) {
    loadingMsg.innerText = "Request error: " + err;
  }
}

document.getElementById("question").addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});
</script>
</body>
</html>
"""


@app.get("/upload-jobs/{job_id}")
def upload_job_status(job_id: str):
    job = get_upload_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")

    return job


# =========================
# PGVECTOR
# =========================

def pg_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    inserted_total = 0
    results = []

    conn = get_pg_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                for saved_file in saved_files:
                    file_name = saved_file["original_name"]
                    file_path = saved_file["file_path"]
                    update_upload_job(
                        job_id,
                        current_file=file_name,
                        message=f"Extracting {file_name}.",
                    )

                    s = now()
                    text = extract_pdf_text(file_path)
                    chunks = chunk_text(text)
                    extract_time += elapsed(s)
                    increment_upload_job(
                        job_id,
                        chunks_total=len(chunks),
                        message=f"Embedding {file_name}: 0/{len(chunks)} chunks.",
                    )

                    rows = []
                    embeddings, batch_embed_time = embed_chunks_for_upload(job_id, file_name, chunks)
                    embed_time += batch_embed_time

                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        rows.append(
                            (
                                file_name,
                                idx,
                                chunk,
                                vector_to_sql(emb),
                                vector_to_halfvec_sql(emb),
                            )
                        )

                    s = now()
                    update_upload_job(
                        job_id,
                        message=f"Inserting {file_name} into PGVector.",
                    )

                    if rows:
                        execute_values(
                            cur,
                            f"""
                            INSERT INTO {PG_TABLE}
                            (file_name, chunk_index, content, embedding, embedding_half)
                            VALUES %s
                            """,
                            rows,
                            template="(%s, %s, %s, %s::vector(4096), %s::halfvec(2048))",
                        )

                    db_time += elapsed(s)

                    inserted_total += len(rows)
                    increment_upload_job(
                        job_id,
                        files_done=1,
                        inserted_chunks=len(rows),
                        message=f"Finished {file_name}.",
                    )
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


@app.post("/pg/upload-pdfs", status_code=202)
async def pg_upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("pg", background_tasks, pg_process_upload_files, files)


@app.post("/pg/ask")
def pg_ask(req: AskRequest):
    total_start = now()
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    q_halfvec = vector_to_halfvec_sql(q_emb)
    q_fullvec = vector_to_sql(q_emb)
    candidate_limit = retrieval_candidate_limit()

    s = now()
    conn = get_pg_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH candidates AS (
                    SELECT
                        id,
                        file_name,
                        chunk_index,
                        content,
                        embedding
                    FROM {PG_TABLE}
                    WHERE embedding_half IS NOT NULL
                    ORDER BY embedding_half <=> %s::halfvec(2048)
                    LIMIT %s
                )
                SELECT
                    file_name,
                    chunk_index,
                    content,
                    1 - (embedding <=> %s::vector(4096)) AS similarity
                FROM candidates
                ORDER BY embedding <=> %s::vector(4096)
                LIMIT %s
                """,
                (
                    q_halfvec,
                    max(PG_CANDIDATE_K, candidate_limit),
                    q_fullvec,
                    q_fullvec,
                    candidate_limit,
                ),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    retrieval_time = elapsed(s)

    if not rows or float(rows[0][3]) < SIMILARITY_THRESHOLD:
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time)

    items = [
        {"file_name": r[0], "chunk_index": r[1], "content": r[2], "similarity": float(r[3])}
        for r in rows
    ]

    s = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(s)

    context, sources = build_context_from_items(ranked_items)

    if not context.strip():
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time, rerank_time)

    s = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(s)

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


@app.get("/pg/docs")
def pg_list_documents():
    conn = get_pg_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT file_name, COUNT(*) AS chunks
                FROM {PG_TABLE}
                GROUP BY file_name
                ORDER BY file_name
                """
            )
            rows = cur.fetchall()

        return {"documents": [{"file_name": r[0], "chunks": int(r[1])} for r in rows]}

    finally:
        conn.close()


@app.delete("/pg/clear")
def pg_clear_documents():
    conn = get_pg_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {PG_TABLE}")

        return {"status": "success", "message": "PGVector documents cleared."}

    finally:
        conn.close()


# =========================
# MILVUS
# =========================

def milvus_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    collection = get_milvus_collection()

    inserted_total = 0
    results = []

    for saved_file in saved_files:
        file_name = saved_file["original_name"]
        file_path = saved_file["file_path"]
        update_upload_job(
            job_id,
            current_file=file_name,
            message=f"Extracting {file_name}.",
        )

        s = now()
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)
        extract_time += elapsed(s)
        increment_upload_job(
            job_id,
            chunks_total=len(chunks),
            message=f"Embedding {file_name}: 0/{len(chunks)} chunks.",
        )

        rows = []
        embeddings, batch_embed_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_embed_time

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append(
                {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "content": chunk[:4000],
                    "embedding": emb,
                }
            )

        s = now()
        update_upload_job(
            job_id,
            message=f"Inserting {file_name} into Milvus.",
        )

        for batch in chunk_batches(rows, MILVUS_INSERT_BATCH_SIZE):
            file_names = [x["file_name"] for x in batch]
            chunk_indexes = [x["chunk_index"] for x in batch]
            contents = [x["content"] for x in batch]
            embeddings = [x["embedding"] for x in batch]

            collection.insert([file_names, chunk_indexes, contents, embeddings])

        collection.flush()
        db_time += elapsed(s)

        inserted_total += len(rows)
        increment_upload_job(
            job_id,
            files_done=1,
            inserted_chunks=len(rows),
            message=f"Finished {file_name}.",
        )
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


@app.post("/milvus/upload-pdfs", status_code=202)
async def milvus_upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("milvus", background_tasks, milvus_process_upload_files, files)


@app.post("/milvus/ask")
def milvus_ask(req: AskRequest):
    total_start = now()
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    candidate_limit = retrieval_candidate_limit()

    collection = get_milvus_collection()

    s = now()
    collection.load()

    if MILVUS_INDEX_TYPE == "HNSW":
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
    elif MILVUS_INDEX_TYPE == "IVF_FLAT":
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    else:
        search_params = {"metric_type": "COSINE", "params": {}}

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param=search_params,
        limit=candidate_limit,
        output_fields=["file_name", "chunk_index", "content"],
    )

    retrieval_time = elapsed(s)

    hits = results[0]

    if not hits or float(hits[0].score) < SIMILARITY_THRESHOLD:
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time)

    items = []

    for hit in hits:
        entity = hit.entity
        items.append(
            {
                "file_name": entity.get("file_name"),
                "chunk_index": entity.get("chunk_index"),
                "content": entity.get("content"),
                "similarity": float(hit.score),
            }
        )

    s = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(s)

    context, sources = build_context_from_items(ranked_items)

    if not context.strip():
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time, rerank_time)

    s = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(s)

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


@app.get("/milvus/docs")
def milvus_list_documents():
    connect_milvus()

    if not utility.has_collection(MILVUS_COLLECTION):
        return {"documents": []}

    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    rows = collection.query(
        expr="id >= 0",
        output_fields=["file_name"],
        limit=10000,
    )

    counts = {}

    for r in rows:
        fn = r.get("file_name")
        counts[fn] = counts.get(fn, 0) + 1

    return {"documents": [{"file_name": k, "chunks": v} for k, v in sorted(counts.items())]}


@app.delete("/milvus/clear")
def milvus_clear_documents():
    connect_milvus()

    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)

    return {"status": "success", "message": "Milvus collection cleared."}


# =========================
# ORACLE 26AI
# =========================

def oracle_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    inserted_total = 0
    results = []

    conn = get_oracle_connection()

    try:
        cur = conn.cursor()

        for saved_file in saved_files:
            file_name = saved_file["original_name"]
            file_path = saved_file["file_path"]
            update_upload_job(
                job_id,
                current_file=file_name,
                message=f"Extracting {file_name}.",
            )

            s = now()
            text = extract_pdf_text(file_path)
            chunks = chunk_text(text)
            extract_time += elapsed(s)
            increment_upload_job(
                job_id,
                chunks_total=len(chunks),
                message=f"Embedding {file_name}: 0/{len(chunks)} chunks.",
            )

            rows = []
            embeddings, batch_embed_time = embed_chunks_for_upload(job_id, file_name, chunks)
            embed_time += batch_embed_time

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                rows.append(
                    {
                        "file_name": file_name,
                        "chunk_index": idx,
                        "content": chunk,
                        "embedding": to_oracle_float32_vector(emb),
                    }
                )

            s = now()
            update_upload_job(
                job_id,
                message=f"Inserting {file_name} into Oracle.",
            )

            cur.setinputsizes(
                file_name=oracledb.DB_TYPE_VARCHAR,
                chunk_index=oracledb.DB_TYPE_NUMBER,
                content=oracledb.DB_TYPE_CLOB,
            )

            if rows:
                cur.executemany(
                    f"""
                    INSERT INTO {ORACLE_TABLE}
                    (file_name, chunk_index, content, embedding)
                    VALUES (:file_name, :chunk_index, :content, :embedding)
                    """,
                    rows,
                )

            conn.commit()
            db_time += elapsed(s)

            inserted_total += len(rows)
            increment_upload_job(
                job_id,
                files_done=1,
                inserted_chunks=len(rows),
                message=f"Finished {file_name}.",
            )
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


@app.post("/oracle/upload-pdfs", status_code=202)
async def oracle_upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("oracle", background_tasks, oracle_process_upload_files, files)


@app.post("/oracle/ask")
def oracle_ask(req: AskRequest):
    total_start = now()
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    q_vec = to_oracle_float32_vector(q_emb)
    candidate_limit = retrieval_candidate_limit()

    s = now()
    conn = get_oracle_connection()

    try:
        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT
                file_name,
                chunk_index,
                content,
                1 - VECTOR_DISTANCE(embedding, :q_vec, COSINE) AS similarity
            FROM {ORACLE_TABLE}
            ORDER BY VECTOR_DISTANCE(embedding, :q_vec, COSINE)
            FETCH FIRST {candidate_limit} ROWS ONLY
            """,
            q_vec=q_vec,
        )

        rows_raw = cur.fetchall()

        rows = []

        for r in rows_raw:
            content_value = r[2]

            if hasattr(content_value, "read"):
                content_value = content_value.read()

            rows.append((r[0], r[1], content_value, r[3]))

    finally:
        conn.close()

    retrieval_time = elapsed(s)

    if not rows or float(rows[0][3]) < SIMILARITY_THRESHOLD:
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time)

    items = [
        {"file_name": r[0], "chunk_index": r[1], "content": r[2], "similarity": float(r[3])}
        for r in rows
    ]

    s = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(s)

    context, sources = build_context_from_items(ranked_items)

    if not context.strip():
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time, rerank_time)

    s = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(s)

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


@app.get("/oracle/docs")
def oracle_list_documents():
    conn = get_oracle_connection()

    try:
        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT file_name, COUNT(*) AS chunks
            FROM {ORACLE_TABLE}
            GROUP BY file_name
            ORDER BY file_name
            """
        )

        rows = cur.fetchall()

        return {"documents": [{"file_name": r[0], "chunks": int(r[1])} for r in rows]}

    finally:
        conn.close()


@app.delete("/oracle/clear")
def oracle_clear_documents():
    conn = get_oracle_connection()

    try:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {ORACLE_TABLE}")
        conn.commit()

        return {"status": "success", "message": "Oracle documents cleared."}

    finally:
        conn.close()


# =========================
# QDRANT
# =========================

def qdrant_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    client = ensure_qdrant_collection()

    inserted_total = 0
    results = []

    for saved_file in saved_files:
        file_name = saved_file["original_name"]
        file_path = saved_file["file_path"]
        update_upload_job(
            job_id,
            current_file=file_name,
            message=f"Extracting {file_name}.",
        )

        s = now()
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)
        extract_time += elapsed(s)
        increment_upload_job(
            job_id,
            chunks_total=len(chunks),
            message=f"Embedding {file_name}: 0/{len(chunks)} chunks.",
        )

        points = []
        embeddings, batch_embed_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_embed_time

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "file_name": file_name,
                        "chunk_index": idx,
                        "content": chunk[:8000],
                    },
                )
            )

        s = now()
        update_upload_job(
            job_id,
            message=f"Inserting {file_name} into Qdrant.",
        )

        for batch in chunk_batches(points, QDRANT_INSERT_BATCH_SIZE):
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch,
            )

        db_time += elapsed(s)

        inserted_total += len(points)
        increment_upload_job(
            job_id,
            files_done=1,
            inserted_chunks=len(points),
            message=f"Finished {file_name}.",
        )
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


@app.post("/qdrant/upload-pdfs", status_code=202)
async def qdrant_upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job("qdrant", background_tasks, qdrant_process_upload_files, files)


@app.post("/qdrant/ask")
def qdrant_ask(req: AskRequest):
    total_start = now()
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    candidate_limit = retrieval_candidate_limit()

    client = ensure_qdrant_collection()

    s = now()

    query_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_emb,
        limit=candidate_limit,
        with_payload=True,
    )

    results = query_result.points
    retrieval_time = elapsed(s)

    if not results or float(results[0].score) < SIMILARITY_THRESHOLD:
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time)

    items = []

    for hit in results:
        payload = hit.payload or {}

        items.append(
            {
                "file_name": payload.get("file_name"),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content"),
                "similarity": float(hit.score),
            }
        )

    s = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(s)

    context, sources = build_context_from_items(ranked_items)

    if not context.strip():
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time, rerank_time)

    s = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(s)

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


@app.get("/qdrant/docs")
def qdrant_list_documents():
    client = ensure_qdrant_collection()

    offset = None
    counts = {}

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}
            fn = payload.get("file_name")

            if fn:
                counts[fn] = counts.get(fn, 0) + 1

        if offset is None:
            break

    docs = [{"file_name": k, "chunks": v} for k, v in sorted(counts.items())]

    return {"documents": docs}


@app.delete("/qdrant/clear")
def qdrant_clear_documents():
    client = get_qdrant_client()

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION in names:
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    return {"status": "success", "message": "Qdrant collection cleared."}


# =========================
# ELASTICSEARCH
# =========================

def elasticsearch_process_upload_files(job_id: str, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    client = ensure_es_index()

    inserted_total = 0
    results = []

    for saved_file in saved_files:
        file_name = saved_file["original_name"]
        file_path = saved_file["file_path"]
        update_upload_job(
            job_id,
            current_file=file_name,
            message=f"Extracting {file_name}.",
        )

        s = now()
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)
        extract_time += elapsed(s)
        increment_upload_job(
            job_id,
            chunks_total=len(chunks),
            message=f"Embedding {file_name}: 0/{len(chunks)} chunks.",
        )

        actions = []
        embeddings, batch_embed_time = embed_chunks_for_upload(job_id, file_name, chunks)
        embed_time += batch_embed_time

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            actions.append(
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
            )

        s = now()
        update_upload_job(
            job_id,
            message=f"Inserting {file_name} into Elasticsearch.",
        )

        for batch in chunk_batches(actions, ES_INSERT_BATCH_SIZE):
            helpers.bulk(
                client,
                batch,
                request_timeout=ES_REQUEST_TIMEOUT,
            )

        if actions:
            client.indices.refresh(index=ES_INDEX)

        db_time += elapsed(s)

        inserted_total += len(actions)
        increment_upload_job(
            job_id,
            files_done=1,
            inserted_chunks=len(actions),
            message=f"Finished {file_name}.",
        )
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


@app.post("/elasticsearch/upload-pdfs", status_code=202)
async def elasticsearch_upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    return await enqueue_upload_job(
        "elasticsearch",
        background_tasks,
        elasticsearch_process_upload_files,
        files,
    )


@app.post("/elasticsearch/ask")
def elasticsearch_ask(req: AskRequest):
    total_start = now()
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    search_query, q_emb, rewrite_time, embed_time = prepare_query_embedding(question)
    candidate_limit = retrieval_candidate_limit()

    client = ensure_es_index()

    s = now()

    response = client.search(
        index=ES_INDEX,
        knn={
            "field": "embedding",
            "query_vector": q_emb,
            "k": candidate_limit,
            "num_candidates": max(ES_NUM_CANDIDATES, candidate_limit),
        },
        source=["file_name", "chunk_index", "content"],
        size=candidate_limit,
    )

    hits = response.get("hits", {}).get("hits", [])
    retrieval_time = elapsed(s)

    if not hits or elastic_score_to_cosine(hits[0].get("_score", 0)) < SIMILARITY_THRESHOLD:
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time)

    items = []

    for hit in hits:
        source = hit.get("_source") or {}

        items.append(
            {
                "file_name": source.get("file_name"),
                "chunk_index": source.get("chunk_index"),
                "content": source.get("content"),
                "similarity": elastic_score_to_cosine(hit.get("_score", 0)),
            }
        )

    s = now()
    ranked_items = rerank_chunks(question, items)
    rerank_time = elapsed(s)

    context, sources = build_context_from_items(ranked_items)

    if not context.strip():
        return empty_result(total_start, embed_time, retrieval_time, rewrite_time, rerank_time)

    s = now()
    answer = ask_llm(question, context)
    llm_time = elapsed(s)

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


@app.get("/elasticsearch/docs")
def elasticsearch_list_documents():
    client = get_es_client()

    if not bool(client.indices.exists(index=ES_INDEX)):
        return {"documents": []}

    response = client.search(
        index=ES_INDEX,
        size=0,
        aggs={
            "documents": {
                "terms": {
                    "field": "file_name",
                    "size": 10000,
                }
            }
        },
    )

    buckets = response.get("aggregations", {}).get("documents", {}).get("buckets", [])
    docs = [{"file_name": b.get("key"), "chunks": int(b.get("doc_count", 0))} for b in buckets]

    return {"documents": docs}


@app.delete("/elasticsearch/clear")
def elasticsearch_clear_documents():
    client = get_es_client()

    if bool(client.indices.exists(index=ES_INDEX)):
        client.indices.delete(index=ES_INDEX)

    return {"status": "success", "message": "Elasticsearch index cleared."}


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pg_table": PG_TABLE,
        "pg_candidate_k": PG_CANDIDATE_K,
        "milvus_collection": MILVUS_COLLECTION,
        "milvus_index_type": MILVUS_INDEX_TYPE,
        "oracle_table": ORACLE_TABLE,
        "qdrant_collection": QDRANT_COLLECTION,
        "elasticsearch_index": ES_INDEX,
        "elasticsearch_num_candidates": ES_NUM_CANDIDATES,
        "elasticsearch_url_scheme": ES_URL.split("://", 1)[0],
        "elasticsearch_verify_certs": ES_VERIFY_CERTS,
        "embedding_dim": EMBEDDING_DIM,
        "halfvec_dim": HALFVEC_DIM,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "chunk_min_words": CHUNK_MIN_WORDS,
        "retrieval_candidate_k": RETRIEVAL_CANDIDATE_K,
        "enable_llm_rerank": ENABLE_LLM_RERANK,
        "rerank_max_candidates": RERANK_MAX_CANDIDATES,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "app_host": APP_HOST,
        "app_port": APP_PORT,
        "app_workers": 1,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting uvicorn on %s:%s with workers=1",
        APP_HOST,
        APP_PORT,
    )
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        workers=1,
        log_level=APP_LOG_LEVEL,
    )
