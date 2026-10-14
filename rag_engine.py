"""
PDF RAG engine: per-chat-thread FAISS index (saved to disk) + the rag_tool used by the agent.
"""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vector_indexes"  # one FAISS folder per chat thread

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

_RETRIEVERS: Dict[str, Any] = {}
_META: Dict[str, dict] = {}


def _thread_dir(thread_id: str) -> Path:
    return INDEX_DIR / str(thread_id)


def _make_retriever(store: FAISS):
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def _build_index(file_bytes: bytes, thread_id: str, filename: Optional[str]) -> dict:
    """Blocking work (PDF parsing + embeddings). Always called via asyncio.to_thread."""
    tid = str(thread_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    try:
        docs = PyPDFLoader(temp_path).load()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No readable text found in this PDF (scanned/image PDFs are not supported).")

    store = FAISS.from_documents(chunks, embeddings)
    folder = _thread_dir(tid)
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))

    meta = {"filename": filename or "document.pdf", "documents": len(docs), "chunks": len(chunks)}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    _RETRIEVERS[tid] = _make_retriever(store)
    _META[tid] = meta
    return meta


async def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Index an uploaded PDF for a chat thread (replaces any previous PDF of that thread)."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    return await asyncio.to_thread(_build_index, file_bytes, str(thread_id), filename)


def _load_index(thread_id: str):
    folder = _thread_dir(thread_id)
    if not (folder / "index.faiss").exists():
        return None
    # Safe: these index files are created by this app, not by users.
    store = FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)
    meta_file = folder / "meta.json"
    if meta_file.exists():
        _META[thread_id] = json.loads(meta_file.read_text(encoding="utf-8"))
    return _make_retriever(store)


async def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    tid = str(thread_id)
    if tid not in _RETRIEVERS:
        retriever = await asyncio.to_thread(_load_index, tid)
        if retriever is None:
            return None
        _RETRIEVERS[tid] = retriever
    return _RETRIEVERS[tid]


def thread_document_metadata(thread_id: str) -> dict:
    """Filename / page / chunk info of the PDF indexed for a thread ({} if none)."""
    tid = str(thread_id)
    if tid in _META:
        return _META[tid]
    meta_file = _thread_dir(tid) / "meta.json"
    if meta_file.exists():
        try:
            _META[tid] = json.loads(meta_file.read_text(encoding="utf-8"))
            return _META[tid]
        except (OSError, ValueError):
            pass
    return {}


@tool
async def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Search the PDF uploaded in this chat and return the most relevant passages.
    Use it for any question about the uploaded document.
    """
    thread_id = (config.get("configurable") or {}).get("thread_id")
    retriever = await _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    docs = await retriever.ainvoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "metadata": [d.metadata for d in docs],
        "source_file": thread_document_metadata(thread_id).get("filename"),
    }