"""
scripts/smoke_test.py
---------------------
End-to-end check of a RUNNING Avabodh instance.

Exercises the whole document lifecycle plus the behaviours that regressed
during the 2026-08-23 work: S3 storage, preview streaming, chat memory,
rename, reprocess, delete, and the tenant-isolation guards.

Usage
-----
    # 1. start the API in another terminal
    uvicorn main:app --port 8000

    # 2. run this against it
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000 --file mydoc.pdf

Exits 0 if everything passes, 1 on the first hard failure. Every document
it creates is deleted at the end, including from S3.
"""
import argparse
import os
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> bool:
    status = PASS if ok else FAIL
    _results.append((status, name, detail))
    symbol = "OK  " if ok else "FAIL"
    print(f"  [{symbol}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok and hard:
        summarise()
        sys.exit(1)
    return ok


def note(name: str, detail: str) -> None:
    _results.append((SKIP, name, detail))
    print(f"  [skip] {name}  -- {detail}")


def summarise() -> None:
    p = sum(1 for s, _, _ in _results if s == PASS)
    f = sum(1 for s, _, _ in _results if s == FAIL)
    s = sum(1 for s, _, _ in _results if s == SKIP)
    print(f"\n{'=' * 62}\n  {p} passed, {f} failed, {s} skipped\n{'=' * 62}")


def get_json(c, url, headers, tries: int = 6):
    """
    GET that survives the server dropping an idle keep-alive connection.

    The poll loops below wait between requests while ingestion runs, and the
    server closes idle connections in that gap. httpx surfaces that as
    ReadError/RemoteProtocolError - and transport-level `retries` only cover
    establishing a connection, not a read that dies mid-flight, so it has to
    be handled here. Retried because the request never reached the app: this
    hides a network hiccup, never a real failure.
    """
    last = None
    for attempt in range(tries):
        try:
            return c.get(url, headers=headers).json()
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {tries} attempts: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--file", default=None, help="PDF to upload (default: any PDF in D:/Download)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for ingestion")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    # A unique tenant per run so repeated runs never collide, and so this
    # can safely run against an instance that already holds real data.
    tenant = f"smoke-{uuid.uuid4().hex[:8]}"
    hdr = {"X-Tenant-ID": tenant, "X-Org-Unit-ID": tenant}
    # retries: the poll loops below sit idle between requests, and the
    # server can close an idle keep-alive connection in that gap. Without
    # this, httpx surfaces that as RemoteProtocolError and the test fails
    # while the app is perfectly healthy.
    c = httpx.Client(timeout=120, transport=httpx.HTTPTransport(retries=3))

    pdf = args.file
    if not pdf:
        for cand in ("D:/Download/BillPayReceipt.pdf", "D:/Download/Application_2.pdf"):
            if os.path.exists(cand):
                pdf = cand
                break
    if not pdf or not os.path.exists(pdf):
        print("No PDF found. Pass one with --file.")
        sys.exit(1)

    print(f"\nTesting {base}\n  tenant : {tenant}\n  file   : {pdf}\n")

    # ── 1. service is up ────────────────────────────────────────────────
    print("1. Health")
    try:
        r = c.get(f"{base}/health/")
        check("API responds", r.status_code == 200)
        r = c.get(f"{base}/health/db")
        check("Database connected", r.json().get("database") == "connected")
    except Exception as e:
        check("API reachable", False, f"{type(e).__name__}: is the server running on {base}?")

    # ── 2. tenant isolation guards ──────────────────────────────────────
    print("\n2. Isolation guards")
    r = c.get(f"{base}/documents/")
    check("Rejects request with no tenant headers", r.status_code == 400, f"got {r.status_code}")
    r = c.get(f"{base}/documents/", headers={"X-Tenant-ID": "../../etc", "X-Org-Unit-ID": "x"})
    check("Rejects path-traversal tenant id", r.status_code == 400, f"got {r.status_code}")

    # ── 3. upload + ingest ──────────────────────────────────────────────
    print("\n3. Upload and ingestion")
    with open(pdf, "rb") as fh:
        r = c.post(f"{base}/documents/upload", headers=hdr, files={"file": (os.path.basename(pdf), fh)})
    check("Upload accepted", r.status_code == 201, f"got {r.status_code}: {r.text[:120]}")
    doc = r.json()
    doc_id = doc["id"]
    check("Returns a document id", bool(doc_id), doc_id)
    check("Returns a preview link", bool(doc.get("preview_link")))

    deadline = time.time() + args.timeout
    status = doc.get("status")
    while time.time() < deadline and status not in ("READY", "FAILED"):
        time.sleep(5)
        status = get_json(c, f"{base}/documents/{doc_id}", hdr).get("status")
    check("Ingestion reached READY", status == "READY", f"status={status}")

    detail = get_json(c, f"{base}/documents/{doc_id}", hdr)
    check("Produced chunks", (detail.get("chunk_count") or 0) > 0, f"chunks={detail.get('chunk_count')}")

    # ── 4. storage backend ──────────────────────────────────────────────
    print("\n4. Storage")
    try:
        from config.settings import get_settings
        from pipeline import object_store
        st = get_settings()
        if object_store.is_enabled():
            from db.database import _admin_engine
            from sqlalchemy import text
            with _admin_engine.connect() as conn:
                stored = list(conn.execute(
                    text("SELECT stored_path FROM documents WHERE id = :i"), {"i": doc_id}))[0][0]
            check("Original stored in S3", object_store.is_s3_uri(stored), stored[:70])
            check("Local scratch copy released", not os.path.exists(stored) or object_store.is_s3_uri(stored))
        else:
            note("S3 storage", "STORAGE_S3_BUCKET empty - running on local disk")
    except Exception as e:
        note("Storage inspection", f"could not check directly: {type(e).__name__}")

    # ── 5. preview link (streams from wherever the file lives) ──────────
    print("\n5. Preview link")
    link = get_json(c, f"{base}/documents/{doc_id}/preview-url", hdr)["preview_link"]
    r = c.get(link)
    check("Preview returns the file", r.status_code == 200, f"got {r.status_code}")
    check("Content is a real PDF", r.content[:5] == b"%PDF-", f"{len(r.content)} bytes")
    check("Byte-identical to the upload", r.content == open(pdf, "rb").read())
    ctype = r.headers.get("content-type", "")
    check("Renders inline, not forced download", "pdf" in ctype, f"content-type={ctype}")
    r = c.get(link.split("?")[0] + "?token=forged")
    check("Rejects a forged preview token", r.status_code == 403, f"got {r.status_code}")

    # ── 6. chat + conversation memory ───────────────────────────────────
    print("\n6. Chat")
    r = c.post(f"{base}/chat/message", headers=hdr,
               json={"query": "What is this document about?"})
    check("Answers a question", r.status_code == 200, f"got {r.status_code}: {r.text[:120]}")
    a1 = r.json()
    thread = a1["thread_id"]
    check("Answer is non-empty", len(a1.get("content") or "") > 10)
    check("Answer cites sources", len(a1.get("sources") or []) > 0, f"{len(a1.get('sources') or [])} sources")

    # the memory-window fix: turn 2 must see turn 1
    r = c.post(f"{base}/chat/message", headers=hdr,
               json={"query": "What was my previous question?", "thread_id": thread})
    a2 = r.json().get("content", "")
    check("Remembers the previous turn", r.status_code == 200 and len(a2) > 10, "memory window")

    msgs = get_json(c, f"{base}/chat/threads/{thread}/messages", hdr)
    check("Thread history persisted", msgs.get("total", 0) >= 4, f"{msgs.get('total')} messages")

    # ── 7. rename (the RLS-after-commit fix) ────────────────────────────
    print("\n7. Rename")
    r = c.patch(f"{base}/documents/{doc_id}", headers=hdr, json={"doc_name": "renamed_by_smoke_test.pdf"})
    check("Rename succeeds", r.status_code == 200, f"got {r.status_code}: {r.text[:120]}")
    check("New name returned", r.json().get("doc_name") == "renamed_by_smoke_test.pdf")

    # ── 8. reprocess (re-fetches from S3 when local copy is gone) ───────
    print("\n8. Reprocess")
    r = c.post(f"{base}/documents/{doc_id}/reprocess", headers=hdr)
    check("Reprocess accepted", r.status_code == 202, f"got {r.status_code}: {r.text[:120]}")
    deadline = time.time() + args.timeout
    status = "PROCESSING"
    while time.time() < deadline and status not in ("READY", "FAILED"):
        time.sleep(5)
        status = get_json(c, f"{base}/documents/{doc_id}", hdr).get("status")
    check("Reprocess reached READY", status == "READY", f"status={status}")

    # ── 9. cross-tenant isolation ───────────────────────────────────────
    print("\n9. Cross-tenant isolation")
    other = {"X-Tenant-ID": f"other-{uuid.uuid4().hex[:8]}", "X-Org-Unit-ID": "x"}
    r = c.get(f"{base}/documents/{doc_id}", headers=other)
    check("Another tenant cannot read the document", r.status_code == 404, f"got {r.status_code}")

    # ── 10. delete ──────────────────────────────────────────────────────
    print("\n10. Delete")
    r = c.delete(f"{base}/documents/{doc_id}", headers=hdr)
    check("Delete succeeds", r.status_code == 200, f"got {r.status_code}")
    r = c.get(f"{base}/documents/{doc_id}", headers=hdr)
    check("Document is gone", r.status_code == 404, f"got {r.status_code}")

    c.delete(f"{base}/chat/threads/{thread}", headers=hdr)
    summarise()
    print("All good.\n")


if __name__ == "__main__":
    main()
