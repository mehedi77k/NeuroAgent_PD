import json
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "medical_knowledge" / "pdf_sources"
VECTOR_DIR = BASE_DIR / "data" / "medical_knowledge" / "vector_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 100:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def extract_pdf_chunks():
    all_chunks = []

    for pdf_path in PDF_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_text(text)

            for chunk_index, chunk in enumerate(chunk_text(text)):
                all_chunks.append(
                    {
                        "chunk_id": f"{pdf_path.stem}_p{page_index + 1}_c{chunk_index + 1}",
                        "source_file": pdf_path.name,
                        "page": page_index + 1,
                        "text": chunk,
                    }
                )

    return all_chunks


def build_index():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    chunks = extract_pdf_chunks()

    if not chunks:
        raise RuntimeError("No chunks found. Check PDF folder and text extraction.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

    with open(VECTOR_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"RAG index built successfully.")
    print(f"Chunks: {len(chunks)}")
    print(f"Index path: {VECTOR_DIR / 'index.faiss'}")


if __name__ == "__main__":
    build_index()