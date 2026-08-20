# Merge Summary

This branch was resolved to stay functionally equivalent while removing the remaining merge conflicts against `origin/main`.

## Files Harmonized

- `config/settings.py`
- `pipeline/memory.py`
- `pipeline/retriever.py`
- `requirements.txt`

## What Stayed Intact

- Demo/local env preference still works when `local_demo/.env` exists.
- Blank numeric values in `.env` still fall back safely instead of crashing startup.
- Ollama fallback behavior remains available when `OPENAI_API_KEY` is absent.
- KB chat responses still return structured `sources` data.
- Source citation building still uses the richer metadata path for document references.

## Conflict Resolution Notes

- `config/settings.py` was merged to keep the newer main-branch settings model, plus the demo-env preference and safe blank-port handling.
- `pipeline/memory.py` was aligned to the newer main-branch tenant/org-unit memory behavior while preserving the chat prompt formatting used by the branch.
- `pipeline/retriever.py` was aligned to the newer hybrid-search implementation from main while keeping the shared retrieval contract unchanged.
- `requirements.txt` was restored from the current `main` branch version to remove encoding/binary conflict noise and keep dependency versions stable.

## Verification

- `python -m compileall config pipeline`
- `python -m pytest tests/test_kb.py -q`

Both passed after the merge cleanup.
