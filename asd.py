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

@app.post("/qdrant/upload-pdfs")
async def qdrant_upload_pdfs(files: List[UploadFile] = File(...)):
    total_start = now()
    extract_time = 0
    embed_time = 0
    db_time = 0

    os.makedirs("uploads", exist_ok=True)

    client = ensure_qdrant_collection()

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

        points = []

        for idx, chunk in enumerate(chunks):
            s = now()
            emb = get_embedding(chunk)
            embed_time += elapsed(s)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "file_name": uploaded_file.filename,
                        "chunk_index": idx,
                        "content": chunk[:8000],
                    },
                )
            )

        s = now()

        for batch in chunk_batches(points, QDRANT_INSERT_BATCH_SIZE):
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch,
            )

        db_time += elapsed(s)

        inserted_total += len(points)
        results.append({
            "file": uploaded_file.filename,
            "status": "ok",
            "chunks": len(points),
        })

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


@app.post("/qdrant/ask")
def qdrant_ask(req: AskRequest):
    total_start = now()

    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    s = now()
    q_emb = get_embedding(question)
    embed_time = elapsed(s)

    client = ensure_qdrant_collection()

    s = now()

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q_emb,
        limit=TOP_K,
        with_payload=True,
    )

    retrieval_time = elapsed(s)

    if not results:
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

    top_similarity = float(results[0].score)

    if top_similarity < SIMILARITY_THRESHOLD:
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

    for hit in results:
        payload = hit.payload or {}

        items.append({
            "file_name": payload.get("file_name"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content"),
            "similarity": float(hit.score),
        })

    context, sources = build_context_from_items(items)

    if not context.strip():
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

    s = now()
    answer = ask_llm(question, context)
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

    docs = [
        {"file_name": k, "chunks": v}
        for k, v in sorted(counts.items())
    ]

    return {"documents": docs}


@app.delete("/qdrant/clear")
def qdrant_clear_documents():
    client = get_qdrant_client()

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION in names:
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    return {
        "status": "success",
        "message": "Qdrant collection cleared.",
    }

