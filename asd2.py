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


def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=10,
    )


def get_milvus_collection():
    print("Milvus connect starting", flush=True)
    connect_milvus()
    print("Milvus connected", flush=True)

    if not utility.has_collection(MILVUS_COLLECTION):
        print("Creating Milvus collection only, no index", flush=True)

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

        print("Milvus collection created", flush=True)

    else:
        print("Milvus collection exists", flush=True)
        collection = Collection(MILVUS_COLLECTION)

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
      max-width: 1000px;
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
      margin: 8px 4px 0 4px;
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
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      padding: 9px;
    }
    th {
      background: #f9fafb;
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

  if (value === "pg") {
    document.getElementById("pgTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: PGVector";
  } else {
    document.getElementById("milvusTab").classList.add("active");
    document.getElementById("backendText").innerText = "Active backend: Milvus";
  }

  document.getElementById("uploadStatus").innerText = "";
  document.getElementById("docsBox").innerHTML = "";
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

function getListUrl() {
  return backend === "milvus" ? "/milvus/docs" : "/pg/docs";
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
    listDocs();
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
    listDocs();
  } catch (err) {
    status.innerText = "Request error: " + err;
  }
}

async function listDocs() {
  const box = document.getElementById("docsBox");
  box.innerHTML = "Loading documents...";

  try {
    const res = await fetch(getListUrl());
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
                        results.append({"file": uploaded_file.filename, "status": "no text", "chunks": 0})
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
                    results.append({"file": uploaded_file.filename, "status": "ok", "chunks": len(rows)})

        return {"status": "success", "inserted_chunks": inserted_total, "files": results}

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

    return {"answer": answer, "sources": sources}


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

        docs = [{"file_name": r[0], "chunks": int(r[1])} for r in rows]
        return {"documents": docs}

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
            results.append({"file": uploaded_file.filename, "status": "no text", "chunks": 0})
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
            contents.append(chunk[:4000])
            embeddings.append(emb)

        print("Milvus chunks:", len(chunks), flush=True)
        print("Milvus insert starting", flush=True)

        collection.insert([file_names, chunk_indexes, contents, embeddings])

        print("Milvus insert finished", flush=True)
        print("Milvus flush starting", flush=True)

        collection.flush()

        print("Milvus flush finished", flush=True)

        inserted_total += len(chunks)
        results.append({"file": uploaded_file.filename, "status": "ok", "chunks": len(chunks)})

    return {"status": "success", "inserted_chunks": inserted_total, "files": results}


@app.post("/milvus/create-index")
def milvus_create_index():
    collection = get_milvus_collection()

    index_params = {
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {},
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params,
    )

    return {"status": "success", "message": "Milvus FLAT index created."}


@app.post("/milvus/ask")
def milvus_ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    q_emb = get_embedding(question)

    collection = get_milvus_collection()

    try:
        collection.load()
    except Exception as e:
        logger.error("Milvus load error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Milvus load error: {str(e)}")

    search_params = {
        "metric_type": "COSINE",
        "params": {},
    }

    try:
        results = collection.search(
            data=[q_emb],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["file_name", "chunk_index", "content"],
        )
    except Exception as e:
        logger.error("Milvus search error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Milvus search error. Try POST /milvus/create-index first. Details: {str(e)}",
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

    return {"answer": answer, "sources": sources}


@app.get("/milvus/docs")
def milvus_list_documents():
    connect_milvus()

    if not utility.has_collection(MILVUS_COLLECTION):
        return {"documents": []}

    collection = Collection(MILVUS_COLLECTION)

    try:
        collection.load()
    except Exception:
        pass

    try:
        rows = collection.query(
            expr="id >= 0",
            output_fields=["file_name"],
            limit=10000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query error: {str(e)}")

    counts = {}

    for r in rows:
        fn = r.get("file_name")
        counts[fn] = counts.get(fn, 0) + 1

    docs = [{"file_name": k, "chunks": v} for k, v in sorted(counts.items())]

    return {"documents": docs}


@app.delete("/milvus/clear")
def milvus_clear_documents():
    connect_milvus()

    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)

    return {"status": "success", "message": "Milvus collection cleared."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pg_table": PG_TABLE,
        "milvus_collection": MILVUS_COLLECTION,
    }
