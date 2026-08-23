"""
pipeline/loader.py
------------------
2026-08-21: this file is now /kb-only. The main document pipeline uses
pipeline/extractor.py (unstructured-based) instead — see IMPLEMENTATION_PLAN
(2).md §2.6. load_single_document() is kept here because
pipeline/kb_pipeline.py (the separate, out-of-scope /kb subsystem) imports
it directly and is not part of this migration.

PDF image extraction (extract_images_from_pdf) moved to
pipeline/image_processor.py — it depended on ImageEmbeddingInput/
check_image_hash_exists, both removed from pipeline/embedder.py when chunk
storage moved to Qdrant.
"""

import hashlib
import os
from pathlib import Path
from typing import Generator

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)

LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv":  CSVLoader,
}


def _compute_file_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _enrich_metadata(doc: Document, filepath: str, file_hash: str) -> Document:
    doc.metadata.update({
        "source_path": os.path.abspath(filepath),
        "doc_name":    os.path.basename(filepath),
        "file_hash":   file_hash,
        "file_size":   os.path.getsize(filepath),
    })
    return doc


def load_single_document(filepath: str) -> list[Document]:
    """Load one file and return LangChain Documents."""
    ext = Path(filepath).suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(LOADER_MAP.keys())}"
        )

    loader_cls = LOADER_MAP[ext]

    try:
        loader = loader_cls(filepath)
        docs = loader.load()
        file_hash = _compute_file_hash(filepath)
        enriched = [_enrich_metadata(d, filepath, file_hash) for d in docs]
        logger.info(
            "Loaded '%s' -> %d doc object(s) | hash=%s",
            os.path.basename(filepath), len(enriched), file_hash[:12],
        )
        return enriched
    except Exception as e:
        logger.error("Failed to load '%s': %s", filepath, e)
        raise


def load_directory(directory: str) -> Generator[tuple[str, list[Document]], None, None]:
    """Lazy-load every supported file in a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    supported_files = [
        f for f in directory.rglob("*")
        if f.suffix.lower() in LOADER_MAP and f.is_file()
    ]

    if not supported_files:
        logger.warning("No supported files found in '%s'", directory)
        return

    logger.info("Found %d supported file(s) in '%s'", len(supported_files), directory)

    for filepath in supported_files:
        try:
            docs = load_single_document(str(filepath))
            yield str(filepath), docs
        except Exception as e:
            logger.error("Skipping '%s' due to error: %s", filepath, e)
            continue
