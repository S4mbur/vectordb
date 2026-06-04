
# -*- coding: utf-8 -*-

import os
import uuid
import time
import logging
from typing import List, Dict, Any

import requests
import psycopg2
from psycopg2.extras import execute_values

import oracledb

from fastapi import FastAPI, UploadFile, File, HTTPException
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

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ai_docs_milvus")
MILVUS_INSERT_BATCH_SIZE = int(os.getenv("MILVUS_INSERT_BATCH_SIZE", "16"))

ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_DSN = os.getenv("ORACLE_DSN")
ORACLE_TABLE = os.getenv("ORACLE_TABLE", "AI_DOCS_ORACLE")

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

NOT_FOUND_TR = "Bu bilgi yuklenen dokumanlarda bulunamadi."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triple-rag-app")

app = FastAPI(title="PGVector + Milvus + Oracle 26ai PDF RAG")


class AskRequest(BaseModel):
    question: str


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


def format_oracle_vector(v: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def get_embedding(text: str) -> List[float]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }

    payload = {
        "model": REMOTE_MODEL_NAME,
        "input": text,
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

    data = resp.json()

    try:
        emb = data["data"][0]["embedding"]
    except Exception:
        logger.error("Embedding response: %s", data)
        raise HTTPException(status_code=500, detail="Bad embedding response.")

    if len(emb) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dim mismatch. Expected={EMBEDDING_DIM}, Got={len(emb)}",
        )

    return emb


def ask_llm(question: str, context: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }

    system_prompt = """
You are a strict document-based RAG assistant.
Use only the provided context.
If the context contains enough information to answer, answer clearly in Turkish.
If the answer is not supported by the context, say exactly:
Bu bilgi yuklenen dokumanlarda bulunamadi.

Do not use outside knowledge.
Do not guess.
"""

    payload = {
        "model": REMOTE_CHAT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    }

    resp = requests.post(
        REMOTE_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )

    if resp.status_code != 200:
        logger.error("Chat API error: %s - %s", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Chat API error.")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n--- PAGE {page_no} ---\n{text}")

    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

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


def chunk_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


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

        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "FLAT",
                "params": {},
            },
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
  <title>Triple RAG Chat</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f4f4f5; color: #111827; }
    .container { max-width: 1050px; margin: 35px auto; padding: 20px; }
    .card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom: 20px; }
    .tabs { display: flex; gap: 10px; margin-bottom: 15px; }
    .tab { padding: 12px 18px; border-radius: 10px; border: 1px solid #d1d5db; background: #f9fafb; cursor: pointer; font-weight: 700; }
    .tab.active { background: #111827; color: white; }
    .upload-area { border: 2px dashed #9ca3af; border-radius: 14px; padding: 24px; text-align: center; background: #fafafa; }
    button { border: none; border-radius: 10px; padding: 12px 18px; background: #111827; color: white; cursor: pointer; font-weight: 600; margin: 8px 4px 0 4px; }
    textarea { width: 100%; min-height: 90px; resize: vertical; border-radius: 12px; border: 1px solid #d1d5db; padding: 14px; font-size: 15px; box-sizing: border-box; }
    .chat-box { min-height: 300px; max-height: 540px; overflow-y: auto; background: #f9fafb; border-radius: 14px; padding: 18px; border: 1px solid #e5e7eb; white-space: pre-wrap; }
    .message { padding: 14px; border-radius: 14px; margin-bottom: 12px; line-height: 1.5; }
    .user { background: #e0f2fe; margin-left: 80px; }
    .bot { background: white; border: 1px solid #e5e7eb; margin-right: 80px; }
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

function setBackend(value) {
  backend = value;

  document.getElementById("pgTab").classList.remove("active");
  document.getElementById("milvusTab").classList.remove("active");
  document.getElementById("oracleTab").classList.remove("active");

  if (value === "pg") {
    document.getElementById("pgTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: PGVector";
  } else if (value === "milvus") {
    document.getElementById("milvusTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Milvus";
  } else {
    document.getElementById("oracleTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Oracle 26ai";
  }

  document.getElementById("uploadStatus").innerText = "";
  document.getElementById("docsBox").innerHTML = "";
}

function baseUrl() {
  if (backend === "pg") return "/pg";
  if (backend === "milvus") return "/milvus";
  return "/oracle";
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

async function uploadPdfs() {
  const input = document.getElementById("pdfFiles");
  const status = document.getElementById("uploadStatus");

  if (!input.files.length) {
    status.innerText = "Select at least one PDF.";
    return;
  }

  const formData = new FormData();

  for (const file of input.files) {
    formData.append("files", file);
  }

  status.innerText = "Uploading and embedding...";

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

    status.innerText = "Done. Chunks: " + data.inserted_chunks + timingText(data.timings);
    listDocs();
  } catch (err) {
    status.innerText = "Request error: " + err;
  }
}

async function clearDocs() {
  const status = document.getElementById("uploadStatus");
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
      html += `<tr><td>${d.file_name}</td><td>${d.chunks}</td></tr>`;
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

    let answer = data.answer || "No answer.";

    if (data.sources && data.sources.length > 0) {
      answer += "\\n\\nSources:\\n";
      data.sources.forEach((s, i) => {
        answer += `${i + 1}. ${s.file_name} / chunk ${s.chunk_index} / similarity ${s.similarity.toFixed(4)}\\n`;
      });
    }

    answer += timingText(data.timings);
    loadingMsg.innerText = answer;

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


# =========================
# PGVECTOR
# =========================

@app.post("/pg/upload-pdfs")
async def pg_upload_pdfs(files: List[UploadFile] = File(...)):
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    os.makedirs("uploads", exist_ok=True)
    inserted_total = 0
    results = []

    conn = get_pg_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                for uploaded_file in files:
                    safe_name = f"{uuid.uuid4()}_{uploaded_file.filename}"
                    file_path = os.path.join("uploads", safe_name)

                    with open(file_path, "wb") as f:
                        f.write(await uploaded_file.read())

                    s = now()
                    text = extract_pdf_text(file_path)
                    chunks = chunk_text(text)
                    extract_time += elapsed(s)

                    rows = []

                    for idx, chunk in enumerate(chunks):
                        s = now()
                        emb = get_embedding(chunk)
                        embed_time += elapsed(s)

                        rows.append(
                            (
                                uploaded_file.filename,
                                idx,
                                chunk,
                                vector_to_sql(emb),
                                vector_to_halfvec_sql(emb),
                            )
                        )

                    s = now()
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
                    results.append({"file": uploaded_file.filename, "status": "ok", "chunks": len(rows)})

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


@app.post("/pg/ask")
def pg_ask(req: AskRequest):
    total_start = now()

    s = now()
    q_emb = get_embedding(req.question)
    q_halfvec = vector_to_halfvec_sql(q_emb)
    embed_time = elapsed(s)

    s = now()
    conn = get_pg_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    file_name,
                    chunk_index,
                    content,
                    1 - (embedding_half <=> %s::halfvec(2048)) AS similarity
                FROM {PG_TABLE}
                WHERE embedding_half IS NOT NULL
                ORDER BY embedding_half <=> %s::halfvec(2048)
                LIMIT %s
                """,
                (q_halfvec, q_halfvec, TOP_K),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    retrieval_time = elapsed(s)

    if not rows or float(rows[0][3]) < SIMILARITY_THRESHOLD:
        return {
            "answer": NOT_FOUND_TR,
            "sources": [],
            "timings": {
                "embedding": embed_time,
                "retrieval": retrieval_time,
                "llm": 0,
                "total": elapsed(total_start),
            },
        }

    items = [
        {"file_name": r[0], "chunk_index": r[1], "content": r[2], "similarity": float(r[3])}
        for r in rows
    ]

    context, sources = build_context_from_items(items)

    s = now()
    answer = ask_llm(req.question, context)
    llm_time = elapsed(s)

    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "embedding": embed_time,
            "retrieval": retrieval_time,
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

@app.post("/milvus/upload-pdfs")
async def milvus_upload_pdfs(files: List[UploadFile] = File(...)):
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    os.makedirs("uploads", exist_ok=True)
    collection = get_milvus_collection()

    inserted_total = 0
    results = []

    for uploaded_file in files:
        safe_name = f"{uuid.uuid4()}_{uploaded_file.filename}"
        file_path = os.path.join("uploads", safe_name)

        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())

        s = now()
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)
        extract_time += elapsed(s)

        rows = []

        for idx, chunk in enumerate(chunks):
            s = now()
            emb = get_embedding(chunk)
            embed_time += elapsed(s)

            rows.append(
                {
                    "file_name": uploaded_file.filename,
                    "chunk_index": idx,
                    "content": chunk[:4000],
                    "embedding": emb,
                }
            )

        s = now()

        for batch in chunk_batches(rows, MILVUS_INSERT_BATCH_SIZE):
            file_names = [x["file_name"] for x in batch]
            chunk_indexes = [x["chunk_index"] for x in batch]
            contents = [x["content"] for x in batch]
            embeddings = [x["embedding"] for x in batch]

            collection.insert([file_names, chunk_indexes, contents, embeddings])
            collection.flush()

        db_time += elapsed(s)

        inserted_total += len(rows)
        results.append({"file": uploaded_file.filename, "status": "ok", "chunks": len(rows)})

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


@app.post("/milvus/ask")
def milvus_ask(req: AskRequest):
    total_start = now()

    s = now()
    q_emb = get_embedding(req.question)
    embed_time = elapsed(s)

    collection = get_milvus_collection()

    s = now()
    collection.load()

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=TOP_K,
        output_fields=["file_name", "chunk_index", "content"],
    )

    retrieval_time = elapsed(s)

    hits = results[0]

    if not hits or float(hits[0].score) < SIMILARITY_THRESHOLD:
        return {
            "answer": NOT_FOUND_TR,
            "sources": [],
            "timings": {
                "embedding": embed_time,
                "retrieval": retrieval_time,
                "llm": 0,
                "total": elapsed(total_start),
            },
        }

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

    context, sources = build_context_from_items(items)

    s = now()
    answer = ask_llm(req.question, context)
    llm_time = elapsed(s)

    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "embedding": embed_time,
            "retrieval": retrieval_time,
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

@app.post("/oracle/upload-pdfs")
async def oracle_upload_pdfs(files: List[UploadFile] = File(...)):
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    os.makedirs("uploads", exist_ok=True)

    inserted_total = 0
    results = []

    conn = get_oracle_connection()

    try:
        cur = conn.cursor()

        for uploaded_file in files:
            safe_name = f"{uuid.uuid4()}_{uploaded_file.filename}"
            file_path = os.path.join("uploads", safe_name)

            with open(file_path, "wb") as f:
                f.write(await uploaded_file.read())

            s = now()
            text = extract_pdf_text(file_path)
            chunks = chunk_text(text)
            extract_time += elapsed(s)

            rows = []

            for idx, chunk in enumerate(chunks):
                s = now()
                emb = get_embedding(chunk)
                embed_time += elapsed(s)

                rows.append(
                    {
                        "file_name": uploaded_file.filename,
                        "chunk_index": idx,
                        "content": chunk,
                        "embedding": format_oracle_vector(emb),
                    }
                )

            s = now()

            cur.executemany(
                f"""
                INSERT INTO {ORACLE_TABLE}
                (file_name, chunk_index, content, embedding)
                VALUES (:file_name, :chunk_index, :content, TO_VECTOR(:embedding))
                """,
                rows,
            )

            conn.commit()
            db_time += elapsed(s)

            inserted_total += len(rows)
            results.append({"file": uploaded_file.filename, "status": "ok", "chunks": len(rows)})

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


@app.post("/oracle/ask")
def oracle_ask(req: AskRequest):
    total_start = now()

    s = now()
    q_emb = get_embedding(req.question)
    q_vec = format_oracle_vector(q_emb)
    embed_time = elapsed(s)

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
                1 - VECTOR_DISTANCE(embedding, TO_VECTOR(:q_vec), COSINE) AS similarity
            FROM {ORACLE_TABLE}
            ORDER BY VECTOR_DISTANCE(embedding, TO_VECTOR(:q_vec), COSINE)
            FETCH FIRST :top_k ROWS ONLY
            """,
            q_vec=q_vec,
            top_k=TOP_K,
        )

        rows = cur.fetchall()

    finally:
        conn.close()

    retrieval_time = elapsed(s)

    if not rows or float(rows[0][3]) < SIMILARITY_THRESHOLD:
        return {
            "answer": NOT_FOUND_TR,
            "sources": [],
            "timings": {
                "embedding": embed_time,
                "retrieval": retrieval_time,
                "llm": 0,
                "total": elapsed(total_start),
            },
        }

    items = [
        {"file_name": r[0], "chunk_index": r[1], "content": r[2], "similarity": float(r[3])}
        for r in rows
    ]

    context, sources = build_context_from_items(items)

    s = now()
    answer = ask_llm(req.question, context)
    llm_time = elapsed(s)

    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "embedding": embed_time,
            "retrieval": retrieval_time,
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
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pg_table": PG_TABLE,
        "milvus_collection": MILVUS_COLLECTION,
        "oracle_table": ORACLE_TABLE,
    }
