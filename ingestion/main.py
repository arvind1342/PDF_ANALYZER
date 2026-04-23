# ============================================================
# 📥 INGESTION SERVICE  —  Port 8001
# ============================================================
# Job: Take a PDF → extract text → chunk it → embed it → store it
# ============================================================

import os
import uuid
import io
import pdfplumber
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Ingestion Service")

# ── Connect to ChromaDB ──────────────────────────────────────
# When running locally: host=localhost
# When running in Docker: host comes from environment variable
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=int(os.getenv("CHROMA_PORT", 8004))
)
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# ── Load the embedding model ─────────────────────────────────
# This downloads ~90MB the first time. Subsequent runs use cache.
# "all-MiniLM-L6-v2" is small, fast, and accurate for semantic search.
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")


def extract_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    full_text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {i+1}]\n{text}"
    return full_text


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.
    
    size=500 means each chunk is ~500 words.
    overlap=100 means consecutive chunks share 100 words.
    Overlap prevents losing context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        chunks.append(chunk)
        start += size - overlap
    return chunks


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Main endpoint: receive PDF, process it, store in vector DB."""
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    pdf_id = str(uuid.uuid4())  # Unique ID for this PDF
    file_bytes = await file.read()

    # 1. Extract text
    print(f"Extracting text from {file.filename}...")
    text = extract_text(file_bytes)
    if not text.strip():
        raise HTTPException(400, "No text found. Is the PDF scanned/image-based?")

    # 2. Chunk
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed
    print("Creating embeddings...")
    embeddings = embedding_model.encode(chunks).tolist()

    # 4. Store in ChromaDB
    print("Storing in ChromaDB...")
    collection.add(
        ids=[f"{pdf_id}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{
            "pdf_id": pdf_id,
            "filename": file.filename,
            "chunk_index": i
        } for i in range(len(chunks))]
    )

    return {
        "pdf_id": pdf_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": "PDF processed successfully!"
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}