# AvabodhAI — Setup

New machine, nothing installed. Do these 8 steps in order.

CTRL + CLICK on underlined links.

**Diagrams:** [As-Built](https://claude.ai/code/artifact/afcbb9fd-e56d-4d77-8097-6ed090416e0d) · [Vectors](https://claude.ai/code/artifact/8d217083-cf15-421b-8b5c-c5e4e5a76a47) · [Punch List](https://claude.ai/code/artifact/804eb86b-0c2c-41ff-bb25-0e90e9d83c3c)

---

## 1. Install Postgres

**Windows** → [postgresql.org/download/windows](https://www.postgresql.org/download/windows/)
Run the installer. Remember the password you set — you'll put it in `.env` as `DB_PASSWORD` in step 6. (`postgres123` is what the app assumes if you don't change anything.)

**macOS**
```bash
brew install postgresql@16
brew services start postgresql@16
psql postgres -c "ALTER USER postgres PASSWORD 'postgres123';"
```

**Linux**
```bash
sudo apt-get install -y postgresql
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres123';"
```

---

## 2. Install Qdrant

Easiest is Docker ([get Docker here](https://www.docker.com/get-started/)):

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage --restart unless-stopped qdrant/qdrant
```

No Docker? Download a binary from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases) and run it.

Check it's up:
```bash
curl http://localhost:6333/healthz
```

---

## 3. Install Python 3.11

[python.org/downloads](https://www.python.org/downloads/) — on Windows, **tick "Add Python to PATH"**.

Not 3.12+. Some packages aren't ready for it.

---

## 4. Install system tools for reading PDFs

These are programs, not Python packages. Python can't install them.

**Windows** — use winget. It installs and sets the PATH for you:

```powershell
winget install oschwartz10612.Poppler
winget install UB-Mannheim.TesseractOCR
```

Close and reopen your terminal, then check:

```powershell
pdfinfo -v
tesseract --version
```

Known-good versions — these are the latest, and what the system is developed
against: **Poppler 25.07.0**, **Tesseract 5.4.0**.

> Use the exact IDs above. There are other Tesseract packages on winget,
> including one last updated in 2014.

No winget? Download manually and add each `bin` folder to your PATH:
[Poppler](https://github.com/oschwartz10612/poppler-windows/releases) ·
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

**macOS**
```bash
xcode-select --install
brew install libmagic poppler tesseract
```

**Linux (Ubuntu / Debian)**
```bash
sudo apt-get update
sudo apt-get install -y   gcc g++   libpq-dev   libmagic1   poppler-utils   tesseract-ocr   libgl1 libglib2.0-0 libsm6 libxext6
```

| Package | Needed for |
|---|---|
| `gcc`, `g++` | compiling a few Python packages during install |
| `libpq-dev` | talking to Postgres |
| `libmagic1` | detecting what type an uploaded file really is |
| `poppler-utils` | rendering PDF pages |
| `tesseract-ocr` | reading text out of scanned documents |
| `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6` | the layout model that finds tables and charts on a page |

Skip any of these and PDF uploads fail with confusing errors.

---

## 5. Install the app

```bash
python -m venv .venv
```

Activate — **every new terminal**:
```bash
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

**Install the Python packages.** Install torch FIRST — otherwise pip pulls the
CUDA build (~2 GB) and nothing here uses a GPU:

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Takes 5-15 minutes.

**Download the AI models that run on your machine** (~600 MB, once). Skipping
this doesn't break anything immediately — it just moves the download into your
first real request, which will appear to hang for minutes:
```bash
python -m nltk.downloader punkt_tab punkt
python -c "from fastembed import SparseTextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1'); TextCrossEncoder(model_name='BAAI/bge-reranker-base')"
python -c "from unstructured_inference.models.base import get_model; get_model()"
python -c "from unstructured_inference.models.tables import load_agent; load_agent()"
```

| Command | Downloads |
|---|---|
| `nltk.downloader` | Sentence splitters used when reading DOCX files |
| `fastembed` line | The keyword-search model and the result re-ranker |
| `get_model()` | The model that finds tables and charts on a PDF page |
| `load_agent()` | The model that reads a table's rows and columns |

**Only if you need website scraping** (the `/web/scrape` endpoint):
```bash
playwright install-deps chromium    # Linux only
playwright install chromium
```

---

## 6. Configure

```bash
cp .env.example .env
```

Everything the app needs comes from environment variables, read from this
`.env` file. **Change any of them to match your own machine** — nothing is
hardcoded.

These have no useful default, so you must set them:

```bash
OPENAI_API_KEY=sk-your-key-here
SECRET_KEY=any-long-random-string

# The public address this deployment is reached at. Preview links are
# built from it. Blank is fine on your own machine; NOT fine once it is
# behind a domain — see "Deploying to a real domain" below.
PUBLIC_BASE_URL=http://localhost:8000
```

The rest already point at a standard local install. Override whichever don't
match yours:

| Variable | Default | Change it if... |
|---|---|---|
| `DB_HOST` | `localhost` | Postgres is on another machine |
| `DB_PORT` | `5432` | you changed the port during install |
| `DB_NAME` | `Avabodh` | you want a different name (note the capital A) |
| `DB_USER` | `postgres` | your admin user isn't `postgres` |
| `DB_PASSWORD` | `postgres123` | **you set a different password in step 1** |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant is elsewhere |
| `QDRANT_API_KEY` | *(empty)* | your Qdrant requires a key |
| `APP_DB_USER` | `avabodh_app` | you want a different name for the app's own user |
| `APP_DB_PASSWORD` | `CHANGE-ME-...` | **always, before production** |
| `PUBLIC_BASE_URL` | *(empty)* | **you deploy behind a domain** — see below |

> If you used a different Postgres password in step 1, set `DB_PASSWORD` here
> and everything downstream follows automatically. The setup script and the
> app both read these same variables — there is nowhere else to change them.

### Deploying to a real domain

Three settings must differ **per environment** — dev and production cannot
share them:

| Variable | Why |
|---|---|
| `PUBLIC_BASE_URL` | Preview links are built from it. Blank works locally, but behind a proxy it guesses the container's internal address and every link becomes unreachable. Set `https://dev.v2.clariona.ai` / `https://v2.clariona.ai`. |
| `SECRET_KEY` | Signs those preview links. Share it and a link issued by dev validates against production. |
| `STORAGE_S3_KEY_PREFIX` | Keeps each environment's files in its own folder in the bucket, so wiping dev can't touch production. |

The S3 object path itself contains no hostname, so moving between domains
never breaks stored files.


---

## 7. Create the database

```bash
python scripts/init_schema.py --create-database
python scripts/init_qdrant.py
```

**What the first one does** (using the `DB_*` variables from step 6):
- Connects as `DB_USER` with `DB_PASSWORD`
- Creates the `DB_NAME` database
- Creates 5 tables: `documents`, `chat_threads`, `chat_messages`, `kb_documents`, `kb_chat_history`
- Creates a second, restricted user `avabodh_app` — **the app runs as this one, not as `postgres`**
- Turns on row-level security so one customer can't see another's data

Why the second user: security rules **don't apply to admin accounts**. Run the app as your admin user and isolation silently stops working.

Your admin user needs `CREATEDB` and `CREATEROLE` privileges for this one-time
setup. The app never uses that account afterwards.

Both scripts are safe to re-run. Drop `--create-database` after the first time.

---

## 8. Run it

```bash
uvicorn main:app --reload --port 8000
```

Wait for `Application startup complete.` — boot takes ~15 seconds because it loads all the AI models up front, so no request has to wait.

Open **http://localhost:8000/docs**

---

## Test it

```bash
curl http://localhost:8000/health/db
```

Upload a PDF:
```bash
curl -X POST http://localhost:8000/documents/upload -H "X-Tenant-ID: test" -H "X-Org-Unit-ID: test" -F "file=@yourfile.pdf"
```

Copy the `id` from the response, then check until `status` is `READY`:
```bash
curl http://localhost:8000/documents/PASTE-ID -H "X-Tenant-ID: test" -H "X-Org-Unit-ID: test"
```

Ask a question:
```bash
curl -X POST http://localhost:8000/chat/message -H "Content-Type: application/json" -H "X-Tenant-ID: test" -H "X-Org-Unit-ID: test" -d "{\"query\":\"What is this about?\"}"
```

Done.

---

## Running with Docker instead

Skip steps 3, 4, 5 and 8. Do steps 1, 2, 6, then:

```bash
docker compose build
docker compose up -d
docker compose exec api python scripts/init_schema.py --create-database
docker compose exec api python scripts/init_qdrant.py
docker compose logs -f api
```

**`.env` is ignored in Docker.** The container reads real environment
variables, and `docker-compose.yml` decides which ones reach it. Anything you
tune in `.env` must also be set in your deployment environment, or it silently
falls back to the code default.

This is easy to get wrong and hard to notice. A real example from this repo:
`SEARCH_SCORE_THRESHOLD` was tuned to `0.35` in `.env` but was never passed
through `docker-compose.yml` — so containers ran with **no relevance floor at
all** and returned different search results than a dev machine, with nothing
in the logs to say so. Every setting is now passed through; if you add a new
one, add it there too.

---

## If something breaks

| Problem | Fix |
|---|---|
| Won't start, mentions API key | Set `OPENAI_API_KEY` in `.env` |
| Every request returns 400 | You forgot the `X-Tenant-ID` and `X-Org-Unit-ID` headers |
| PDF upload fails | Step 4 tools missing |
| Database connection refused | Postgres isn't running, or wrong password |
| Can't reach Qdrant | Qdrant isn't running |
| First request hangs for minutes | Step 5 model downloads were skipped |
| Search finds nothing | Document isn't `READY` yet, or different tenant headers than you uploaded with |
| Charts give wrong numbers after restart | Docker only — the `visual_crops` volume is missing |

Logs are in `avabodh.log`.

---

## Commands you'll reuse

```bash
source .venv/bin/activate                # every session
uvicorn main:app --reload --port 8000    # run
python scripts/init_schema.py            # re-check database
python scripts/init_qdrant.py            # re-check search index
pytest tests/                            # tests

docker start qdrant                      # restart Qdrant
docker compose up -d                     # Docker: start
docker compose down                      # Docker: stop
docker compose logs -f api               # Docker: logs
```
