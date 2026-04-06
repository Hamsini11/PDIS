import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─── CONFIG ──────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, already in your venv
STORE_DIR   = Path("storage")
INDEX_FILE  = STORE_DIR / "faiss.index"
META_FILE   = STORE_DIR / "chunks.json"

STORE_DIR.mkdir(exist_ok=True)

# ─── INIT ─────────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
DIM = 384  # all-MiniLM-L6-v2 output dimension

def _load_or_create_index():
    if INDEX_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    return faiss.IndexFlatL2(DIM)

def _load_chunks() -> list[dict]:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return []

def _save(index, chunks):
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(chunks, indent=2))

# ─── CHUNKING ─────────────────────────────────────────────────
def _chunk_page(page_text: str, page_num: int, doc_name: str,
                chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split page text into overlapping chunks with metadata."""
    words = page_text.split()
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append({
                "text":      chunk_text,
                "doc":       doc_name,
                "page":      page_num,
                "chunk_idx": chunk_idx,
                "id":        f"{doc_name}::p{page_num}::c{chunk_idx}"
            })
        i += chunk_size - overlap
        chunk_idx += 1
    return chunks

# ─── PUBLIC API ───────────────────────────────────────────────
def index_document(page_texts, dates: list, filename, sections: list = None):
    """
    Embed and store all page chunks + structured date metadata.
    Call this after run_extraction() in main.py.
    """
    index  = _load_or_create_index()
    chunks = _load_chunks()

    # remove existing entries for this doc (re-indexing)
    chunks = [c for c in chunks if c["doc"] != filename]

    new_chunks = []

    # 1. index page text chunks
    for page_num, page_text in page_texts:
        if not page_text.strip():
            continue
        page_chunks = _chunk_page(page_text, page_num, filename)
        new_chunks.extend(page_chunks)

    # 2. also index each date as its own searchable chunk
    for d in dates:
        date_text = (
            f"Date found: {d["raw"]}. "
            f"Normalized: {d["normalized"] or 'N/A'}. "
            f"Context: {d["context"]}. "
            f"Page: {d["page"]}. "
            f"Document: {filename}."
        )
        new_chunks.append({
            "text":      date_text,
            "doc":       filename,
            "page":      d["page"],
            "chunk_idx": "date",
            "id":        f"{filename}::date::{d["raw"]}::{d["page"]}",
            "is_date":   True,
            "raw_date":  d["raw"],
            "normalized": d["normalized"],
            "confidence": d["confidence"],
            "ambiguous":  d["ambiguous"],
            "flagged":    d["confidence"] < 0.7 or d["ambiguous"]
        })

    if not new_chunks:
        print("  No chunks to index.")
        return

    # embed
    texts = [c["text"] for c in new_chunks]
    print(f"  Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts, show_progress_bar=False,
                              batch_size=32, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)

    index.add(vectors)
    chunks.extend(new_chunks)
    _save(index, chunks)

    sections_file = STORE_DIR / "sections.json"
    existing = {}
    if sections_file.exists():
        try:
            existing = json.loads(sections_file.read_text())
        except Exception:
            existing = {}
    existing[filename] = sections
    sections_file.write_text(json.dumps(existing, indent=2))

    print(f"  ✅ Indexed {len(new_chunks)} chunks "
          f"({len(dates)} date entries) for {filename}")


def search(query: str, top_k: int = 5,
           filter_doc: str = None,
           only_dates: bool = False) -> list[dict]:
    """
    Semantic search over all indexed chunks.
    filter_doc: restrict to a specific document
    only_dates: return only date-type chunks
    """
    index  = _load_or_create_index()
    chunks = _load_chunks()

    if index.ntotal == 0 or not chunks:
        return []

    q_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_vec, min(top_k * 3, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        if filter_doc and chunk["doc"] != filter_doc:
            continue
        if only_dates and not chunk.get("is_date"):
            continue
        results.append({**chunk, "score": float(dist)})
        if len(results) >= top_k:
            break

    return results


def get_all_dates(filter_doc: str = None,
                  flagged_only: bool = False) -> list[dict]:
    """
    Retrieve all indexed dates without semantic search.
    Used by the structured query router.
    """
    chunks = _load_chunks()
    dates = [c for c in chunks if c.get("is_date")]
    if filter_doc:
        dates = [d for d in dates if d["doc"] == filter_doc]
    if flagged_only:
        dates = [d for d in dates if d.get("flagged")]
    return dates


def list_documents() -> list[str]:
    """Return all unique document names in the index."""
    chunks = _load_chunks()
    return list({c["doc"] for c in chunks})


def clear_index():
    """Wipe the index — useful for re-indexing everything."""
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if META_FILE.exists():
        META_FILE.unlink()
    print("Index cleared.")