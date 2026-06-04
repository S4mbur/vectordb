# -*- coding: utf-8 -*-

import os
import uuid
import logging
from typing import List

import requests
import psycopg2
from psycopg2.extras import execute_values

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

PG_HOST = os.getenv("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DBNAME = os.getenv("PG_DBNAME")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE = os.getenv("PG_TABLE", "ai_docs")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ai_docs_milvus")

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
logger = logging.getLogger("dual-rag-app")

app = FastAPI(title="PGVector + Milvus PDF RAG")


class AskRequest(BaseModel):
    question: str


def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def vector_to_sql(v: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def vector_to_halfvec_sql(v: List[float]) -> str:
    half = v[:HALFVEC_DIM]
    return "[" + ",".join(str(float(x)) for x in half) + "]"


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
        timeout=90,
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
    if not REMOTE_CHAT_URL:
        raise HTTPException(status_code=500, detail="REMOTE_CHAT_URL missing.")

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
        timeout=120,
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


def build_context_from_items(items):
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


def get_milvus_collection():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=16000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="PDF RAG collection",
        )

        collection = Collection(
            name=MILVUS_COLLECTION,
            schema=schema,
        )

        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 64,
            },
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
    else:
        collection = Collection(MILVUS_COLLECTION)

    collection.load()
    return collection


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Dual RAG Chat</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f4f5;
      color: #111827;
    }
    .container {
      max-width: 950px;
      margin: 35px auto;
      padding: 20px;
    }
    .card {
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08);
      margin-bottom: 20px;
    }
    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 15px;
    }
    .tab {
      padding: 12px 18px;
      border-radius: 10px;
      border: 1px solid #d1d5db;
      background: #f9fafb;
      cursor: pointer;
      font-weight: 700;
    }
    .tab.active {
      background: #111827;
      color: white;
    }
    .upload-area {
      border: 2px dashed #9ca3af;
      border-radius: 14px;
      padding: 24px;
      text-align: center;
      background: #fafafa;
    }
    button {
      border: none;
      border-radius: 10px;
      padding: 12px 18px;
      background: #111827;
      color: white;
      cursor: pointer;
      font-weight: 600;
      margin-top: 12px;
    }
    textarea {
      width: 100%;
      min-height: 90px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid #d1d5db;
      padding: 14px;
      font-size: 15px;
      box-sizing: border-box;
    }
    .chat-box {
      min-height: 300px;
      max-height: 540px;
      overflow-y: auto;
      background: #f9fafb;
      border-radius: 14px;
      padding: 18px;
      border: 1px solid #e5e7eb;
      white-space: pre-wrap;
    }
    .message {
      padding: 14px;
      border-radius: 14px;
      margin-bottom: 12px;
      line-height: 1.5;
    }
    .user {
      background: #e0f2fe;
      margin-left: 80px;
    }
    .bot {
      background: white;
      border: 1px solid #e5e7eb;
      margin-right: 80px;
    }
    .status {
      font-size: 14px;
      color: #6b7280;
      margin-top: 10px;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: flex-start;
    }
    .row textarea {
      flex: 1;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>PDF RAG Chat</h1>

      <div class="tabs">
        <div id="pgTab" class="tab active" onclick="setBackend('pg')">PGVector</div>
        <div id="milvusTab" class="tab" onclick="setBackend('milvus')">Milvus</div>
      </div>

      <div class="upload-area">
        <strong id="backendText">Active backend: PGVector</strong><br />
        <input id="pdfFiles" type="file" accept="application/pdf" multiple />
        <br />
        <button onclick="uploadPdfs()">Upload PDFs</button>
        <button onclick="clearDocs()">Clear Active DB</button>
        <div id="uploadStatus" class="status"></div>
      </div>
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

  if (value === "pg") {
    document.getElementById("pgTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: PGVector";
  } else {
    document.getElementById("milvusTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Milvus";
  }

  document.getElementById("uploadStatus").innerText = "";
}

function getUploadUrl() {
  return backend === "milvus" ? "/milvus/upload-pdfs" : "/pg/upload-pdfs";
}

function getAskUrl() {
  return backend === "milvus" ? "/milvus/ask" : "/pg/ask";
}

function getClearUrl() {
  return backend === "milvus" ? "/milvus/clear" : "/pg/clear";
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
    const res = await fetch(getUploadUrl(), {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      status.innerText = "Error: " + JSON.stringify(data);
      return;
    }

    status.innerText = "Done. Chunks: " + data.inserted_chunks;
  } catch (err) {
    status.innerText = "Request error: " + err;
  }
}

async function clearDocs() {
  const status = document.getElementById("uploadStatus");
  status.innerText = "Clearing...";

  try {
    const res = await fetch(getClearUrl(), { method: "DELETE" });
    const data = await res.json();

    if (!res.ok) {
      status.innerText = "Error: " + JSON.stringify(data);
      return;
    }

    status.innerText = data.message;
  } catch (err) {
    status.innerText = "Request error: " + err;
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
    const res = await fetch(getAskUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
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


@app.post("/pg/upload-pdfs")
async def pg_upload_pdfs(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 PDF allowed.")

    os.makedirs("uploads", exist_ok=True)

    inserted_total = 0
    results = []

    conn = get_pg_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                for uploaded_file in files:
                    if not uploaded_file.filename.lower().endswith(".pdf"):
                        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

                    safe_name = f"{uuid.uuid4()}_{uploaded_file.filename}"
                    file_path = os.path.join("uploads", safe_name)

                    content_bytes = await uploaded_file.read()

                    with open(file_path, "wb") as f:
                        f.write(content_bytes)

                    text = extract_pdf_text(file_path)

                    if not text.strip():
                        results.append(
                            {
                                "file": uploaded_file.filename,
                                "status": "no text",
                                "chunks": 0,
                            }
                        )
                        continue

                    chunks = chunk_text(text)

                    rows = []

                    for idx, chunk in enumerate(chunks):
                        emb = get_embedding(chunk)

                        rows.append(
                            (
                                uploaded_file.filename,
                                idx,
                                chunk,
                                vector_to_sql(emb),
                                vector_to_halfvec_sql(emb),
                            )
                        )

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

                    inserted_total += len(rows)

                    results.append(
                        {
                            "file": uploaded_file.filename,
                            "status": "ok",
                            "chunks": len(rows),
                        }
                    )

        return {
            "status": "success",
            "inserted_chunks": inserted_total,
            "files": results,
        }

    finally:
        conn.close()


@app.post("/pg/ask")
def pg_ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    q_emb = get_embedding(question)
    q_halfvec = vector_to_halfvec_sql(q_emb)

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

    if not rows:
        return {"answer": NOT_FOUND_TR, "sources": []}

    top_similarity = float(rows[0][3])

    if top_similarity < SIMILARITY_THRESHOLD:
        return {"answer": NOT_FOUND_TR, "sources": []}

    items = [
        {
            "file_name": r[0],
            "chunk_index": r[1],
            "content": r[2],
            "similarity": float(r[3]),
        }
        for r in rows
    ]

    context, sources = build_context_from_items(items)

    if not context.strip():
        return {"answer": NOT_FOUND_TR, "sources": []}

    answer = ask_llm(question, context)

    return {
        "answer": answer,
        "sources": sources,
    }


@app.delete("/pg/clear")
def pg_clear_documents():
    conn = get_pg_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {PG_TABLE}")

        return {
            "status": "success",
            "message": "PGVector documents cleared.",
        }

    finally:
        conn.close()


@app.post("/milvus/upload-pdfs")
async def milvus_upload_pdfs(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 PDF allowed.")

    os.makedirs("uploads", exist_ok=True)

    collection = get_milvus_collection()

    inserted_total = 0
    results = []

    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        safe_name = f"{uuid.uuid4()}_{uploaded_file.filename}"
        file_path = os.path.join("uploads", safe_name)

        content_bytes = await uploaded_file.read()

        with open(file_path, "wb") as f:
            f.write(content_bytes)

        text = extract_pdf_text(file_path)

        if not text.strip():
            results.append(
                {
                    "file": uploaded_file.filename,
                    "status": "no text",
                    "chunks": 0,
                }
            )
            continue

        chunks = chunk_text(text)

        file_names = []
        chunk_indexes = []
        contents = []
        embeddings = []

        for idx, chunk in enumerate(chunks):
            emb = get_embedding(chunk)

            file_names.append(uploaded_file.filename)
            chunk_indexes.append(idx)
            contents.append(chunk[:15000])
            embeddings.append(emb)

        collection.insert(
            [
                file_names,
                chunk_indexes,
                contents,
                embeddings,
            ]
        )

        collection.flush()

        inserted_total += len(chunks)

        results.append(
            {
                "file": uploaded_file.filename,
                "status": "ok",
                "chunks": len(chunks),
            }
        )

    return {
        "status": "success",
        "inserted_chunks": inserted_total,
        "files": results,
    }


@app.post("/milvus/ask")
def milvus_ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    q_emb = get_embedding(question)

    collection = get_milvus_collection()

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "ef": 100,
        },
    }

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["file_name", "chunk_index", "content"],
    )

    hits = results[0]

    if not hits:
        return {"answer": NOT_FOUND_TR, "sources": []}

    top_similarity = float(hits[0].score)

    if top_similarity < SIMILARITY_THRESHOLD:
        return {"answer": NOT_FOUND_TR, "sources": []}

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

    if not context.strip():
        return {"answer": NOT_FOUND_TR, "sources": []}

    answer = ask_llm(question, context)

    return {
        "answer": answer,
        "sources": sources,
    }


@app.delete("/milvus/clear")
def milvus_clear_documents():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)

    return {
        "status": "success",
        "message": "Milvus collection cleared.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pg_table": PG_TABLE,
        "milvus_collection": MILVUS_COLLECTION,
      }
