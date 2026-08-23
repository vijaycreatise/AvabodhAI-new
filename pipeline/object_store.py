"""
pipeline/object_store.py
------------------------
S3 storage for the ORIGINAL uploaded document.

2026-08-23. Before this, an upload was written to local disk and stayed
there forever — meaning the container was the only copy of a user's file.
Replace or rebuild it and every original is gone: preview links
(GET /documents/{id}/file) break permanently and POST
/documents/{id}/reprocess can never run again. S3 becomes the durable
copy; the server keeps a file only while it is actively parsing or
serving it.

SCOPE — originals only. Deliberately NOT handled here:

  - Visual crops (settings.VISUAL_CROPS_DIR). Those are read on EVERY
    answer that includes a chart, up to MAX_ATTACHED_CROPS per question —
    on S3 that is a network round trip per crop on the hottest path. They
    are also derived data: re-ingesting regenerates them byte for byte.
    An original cannot be regenerated, which is the asymmetry that
    decides which one needs remote durability.
  - Web-scraped .md files (api/routes/web.py). A web document's
    source_path IS the URL, so _build_preview_link returns that URL and
    the .md is never served to anyone — it is transient input to
    ingestion only.

THE SWITCH: settings.STORAGE_S3_BUCKET empty (the default) means every
function here no-ops and callers keep their existing local-disk path.
boto3 is imported lazily INSIDE the functions specifically so a
bucket-less deployment never loads it at all.

Conventions (client construction, presigned-expiry shape) follow
clariona-core/src/storage/s3/client.py so the two services stay
recognisable to each other.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_URI_SCHEME = "s3://"


def is_enabled() -> bool:
    """True when a bucket is configured. The single on/off switch."""
    return bool(settings.STORAGE_S3_BUCKET)


def is_s3_uri(path: Optional[str]) -> bool:
    """
    Does this documents.stored_path point at S3 rather than local disk?

    stored_path stays the single source of truth and simply gained a
    second form. Anything not starting with 's3://' is a local path,
    exactly as before — which is why documents ingested before this
    existed keep working with no migration or backfill.
    """
    return bool(path) and str(path).startswith(_URI_SCHEME)


def _split_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/a/b/c.pdf' -> ('bucket', 'a/b/c.pdf')"""
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _client():
    # Imported here, not at module scope: with no bucket configured this
    # module is still imported (callers check is_enabled()), and a
    # deployment that never uses S3 shouldn't need boto3 loaded.
    import boto3

    return boto3.client(
        "s3",
        region_name=settings.STORAGE_S3_REGION or None,
        endpoint_url=settings.STORAGE_S3_ENDPOINT_URL or None,
        # Blank credentials fall through to boto3's normal chain
        # (instance profile / IAM role / ~/.aws), which is what a real
        # deployment should use instead of static keys.
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
    )


def build_key(tenant_id: str, org_unit_id: str, document_id: str, ext: str) -> str:
    """
    Object key for one document.

    Tenant/org are path components here for the same reason they are on
    local disk — one prefix per department makes bucket-level policies and
    manual inspection sane. They are NOT a security control: isolation is
    enforced by Postgres RLS and the signed preview token, never by the
    key layout.
    """
    prefix = (settings.STORAGE_S3_KEY_PREFIX or "").strip("/")
    parts = [p for p in (prefix, tenant_id, org_unit_id, f"{document_id}{ext}") if p]
    return "/".join(parts)


def upload_file(local_path: str, key: str) -> str:
    """
    Upload and return the 's3://bucket/key' URI to store as stored_path.

    Deliberately NOT exception-swallowing: the caller
    (api/routes/documents.py::upload_document) must fail the request if
    this raises. Falling back to local silently would leave the caller
    believing their file is durably stored when it is not.
    """
    bucket = settings.STORAGE_S3_BUCKET
    _client().upload_file(local_path, bucket, key)
    uri = f"{_URI_SCHEME}{bucket}/{key}"
    logger.info("Uploaded %s -> %s", local_path, uri)
    return uri


def download_to_temp(uri: str, suffix: str = "") -> Path:
    """
    Fetch an object to a temp file and return its path. THE CALLER IS
    RESPONSIBLE FOR DELETING IT.

    Used by reprocess, where the local copy is normally long gone — which
    is precisely the case S3 makes possible and local-only storage did
    not.
    """
    bucket, key = _split_uri(uri)
    fd, tmp = tempfile.mkstemp(suffix=suffix or Path(key).suffix)
    os.close(fd)
    _client().download_file(bucket, key, tmp)
    logger.info("Downloaded %s -> %s", uri, tmp)
    return Path(tmp)


def open_stream(uri: str):
    """
    Return a streaming body for GET /documents/{id}/file.

    Streamed through this API rather than handed out as a presigned URL,
    on purpose. A presigned redirect would change that endpoint's
    response from a file to a 307 pointing at amazonaws.com — a contract
    change for live callers (clariona-core consumes this API). It would
    also introduce a second credential with its own expiry alongside the
    existing signed token, when one auth model is easier to reason about.
    Revisit if preview bandwidth ever becomes the bottleneck.
    """
    bucket, key = _split_uri(uri)
    body = _client().get_object(Bucket=bucket, Key=key)["Body"]

    # Wrapped in a generator with an explicit close: StreamingResponse
    # drains the body on a successful send, but a client that disconnects
    # mid-download leaves the underlying connection to be reclaimed only
    # when the object is finalised. The finally block returns it to the
    # pool immediately instead.
    def _stream():
        try:
            for chunk in body.iter_chunks(chunk_size=64 * 1024):
                yield chunk
        finally:
            body.close()

    return _stream()


def delete(uri: str) -> None:
    """
    Remove an object. Non-fatal: a failure here leaves an orphaned object
    costing pennies, whereas raising would abort a document deletion the
    user asked for and leave the Postgres row behind. Logged loudly so it
    can be reconciled.
    """
    try:
        bucket, key = _split_uri(uri)
        _client().delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted %s", uri)
    except Exception as e:
        logger.warning("Failed to delete %s from S3 (non-fatal, object orphaned): %s", uri, e)
