# 🏏 CricLLM Runbook

Operational reference for every Python file in this project: what it does, when
you'd touch it, and how to run/debug the system. For setup instructions and
the full config reference, see [README.md](README.md).

## How the pieces fit together

```
run_ingestion.py                              ask.py
      │                                           │
      ▼                                           ▼
   pipeline.py  ───┬─── loader.py          embeddings.py (RETRIEVAL_QUERY)
                   ├─── markdown_structure.py     │
                   ├─── block_parser.py           ▼
                   ├─── semantic_chunker.py   vectorstore.py.query()
                   ├─── embeddings.py (RETRIEVAL_DOCUMENT)  │
                   ├─── cache.py                            ▼
                   └─── vectorstore.py.upsert()      google.genai (generation)
```

Ingestion (`run_ingestion.py`) builds the index once. Asking a question
(`ask.py`, or `app.py`'s `/api/ask` route) queries that index every time.
All three share `config.py`, `cache.py`, `embeddings.py`, and `vectorstore.py`.

---

## Entry points (run these directly)

### `scripts/run_ingestion.py`
Builds/updates the vector index from a Markdown file or directory.
```bash
python3 scripts/run_ingestion.py --input data/icc_rulebook.md
python3 scripts/run_ingestion.py --input data/icc_rulebook.md --force   # ignore file-hash cache
python3 scripts/run_ingestion.py --input data/                         # all .md files in a dir
```
Thin wrapper — just adds `src/` to `sys.path` and calls `cricllm.cli.main()`.

### `scripts/ask.py`
Query-time RAG (CLI): embeds a question, retrieves matching chunks, asks
Gemini to answer only from them.
```bash
python3 scripts/ask.py "How many no-balls make an over invalid?"
python3 scripts/ask.py "..." --top-k 8
```
Contains `_SYSTEM_PROMPT` — the instructions given to the generation model,
tuned for the ICC Laws of Cricket (edit this constant to point at a
different domain/persona). Exits with a clear message (not a crash) if the
vector store is empty — run ingestion first. Prints the answer, then a
`--- Sources ---` list of retrieved section paths and their distances
(there's a commented-out `print(match["content"])` line for dumping the raw
retrieved text when debugging why an answer looks wrong).

### `app.py`
Browser front-end: a small Flask server exposing `POST /api/ask`, which
streams the answer back as Server-Sent Events (not one big JSON blob) —
same retrieval + generation logic as `scripts/ask.py`, both going through
`cricllm.qa.QAEngine.answer_stream()`. Plus a `GET /` route serving
`templates/index.html` (vanilla HTML/CSS/JS, no framework, no build step).
```bash
pip install flask
python app.py
# open http://localhost:5000
```
If you deploy this behind gunicorn, streaming isn't just nicer UX — a
worker's timeout clock resets every time a chunk is sent, so a slow
generation (retries, a long answer) doesn't just sit silent until the whole
request gets killed by `WORKER TIMEOUT`. (This alone didn't fix the real
production incident we hit — see "chromadb hangs inside a live gunicorn
worker" below.)

---

## `src/cricllm/` — library modules

| File | Purpose | Touch it when... |
|---|---|---|
| `config.py` | `Settings` (pydantic) — every tunable knob, loaded from `.env`. `BreakpointThresholdType` literal for semantic chunker. | Adding a new env var / default. |
| `exceptions.py` | `CricLLMError` hierarchy: `EmptyDocumentError`, `InvalidEncodingError`, `CorruptedMarkdownError`, `EmbeddingAPIError`, `VectorStoreError`. | Adding a new failure mode that needs distinct handling upstream. |
| `logging_config.py` | `setup_logging()` — rotating file handler (`logs/cricllm.log`) + console. `get_logger(name)` for child loggers. | Changing log format/rotation policy. |
| `hashing.py` | `sha256_of_text` / `sha256_of_file`. Used for cache keys, chunk IDs, and file-change detection. | Rarely — this is stable, load-bearing plumbing. |
| `token_utils.py` | `count_tokens()` — tiktoken `cl100k_base` proxy for Gemini's real tokenizer (Gemini has no public offline tokenizer). | If you want a more accurate token estimate. |
| `loader.py` | `load_markdown_file()` — reads a file, handling invalid UTF-8 (falls back to `charset-normalizer`), empty files, null bytes, and unbalanced code fences (warns, doesn't fail). | Adding handling for a new kind of corrupted input file. |
| `markdown_structure.py` | `split_by_headers()` — wraps `MarkdownHeaderTextSplitter` to split on `#`..`####`, producing `MarkdownSection` objects with a `header_path` breadcrumb. | Changing which header levels count, or how the breadcrumb is built. |
| `block_parser.py` | `split_into_segments()` — within one header section, separates **atomic** blocks (fenced code, tables, lists, curl commands) from **prose**, via a line-by-line state machine. | Adding detection for a new atomic block type (e.g. blockquotes). |
| `semantic_chunker.py` | The core chunking algorithm: header split → atomic/prose segmentation → optional `SemanticChunker` on oversized prose → greedy token-budget packing (`_pack_segments`) → force-split (`_force_split_oversized`) for anything exceeding Gemini's ~2048-token embed limit. Produces `Document` objects with full metadata (`header_path`, `block_types`, `chunk_hash`, `forced_split`, etc). | Tuning chunk size/target, changing what counts as "oversized," or the packing strategy. |
| `embeddings.py` | `CachedGeminiEmbeddings` — batches texts, retries with exponential backoff (`tenacity`), and caches every successful **batch** immediately (so a mid-run failure doesn't lose already-embedded work). `embed_query()` reuses the same cached path. | Changing batch size behavior, retry policy, or the embedding provider. |
| `cache.py` | SQLite-backed: `EmbeddingCache` (keyed on `content_hash + model + task_type` — task_type matters, see Incident Playbook below), `FileStateStore` (per-file SHA256 for incremental re-runs), `IngestedChunkStore` (durable ledger of which chunk hashes have been written to the vector store — this is what makes resume-after-interruption work). | Changing what gets cached/tracked, or the SQLite schema. |
| `vectorstore.py` | `VectorStore` — thin wrapper over a hosted **Pinecone** index (see the chromadb incident below for why it's not local anymore). Creates the index with cosine metric if it doesn't exist. Chunk content also rides along in Pinecone metadata under `"content"` (Pinecone has nowhere else to put it), unpacked back out in `query()`. Converts Pinecone's similarity score to a `1 - score` "distance" so the rest of the app doesn't care which vector store is behind it. `existing_ids()` / `upsert()` / `query()` / `count()`. | Changing the vector DB backend or query behavior. |
| `models.py` | `IngestionStats` — per-file result summary (chunks ingested/duplicate/skipped, errors). | Adding a new stat to track/report. |
| `pipeline.py` | `IngestionPipeline.ingest_file()` — orchestrates one file end-to-end: file-hash check → load → chunk → dedupe → filter-already-ingested → embed → upsert → mark completed. Writes failed batches to `logs/dead_letter/*.json` instead of losing progress. | Changing the overall ingestion flow/ordering. |
| `cli.py` | `cricllm.cli.main()` — argparse entry point behind `run_ingestion.py`. Walks a file or directory of `.md` files and calls `ingest_file()` on each. | Adding a new CLI flag. |
| `qa.py` | `QAEngine` — the actual "answer a question" logic, shared by `scripts/ask.py` and `app.py` so neither has its own copy of the system prompt or retrieval flow. `answer_stream()` yields `StreamEvent`s: one `kind="sources"` right after retrieval, then a growing series of `kind="delta"` as Gemini streams the answer (each one is the *full* text so far, not just the new bit). | Changing the system prompt, or how retrieval feeds into the generation prompt. |
| `__init__.py` | Just `__version__`. | Bumping version. |

### Project root

| File | Purpose |
|---|---|
| `app.py` | Flask web UI — `POST /api/ask` (streams the answer as Server-Sent Events) and `GET /` (serves `templates/index.html`). Not part of the `cricllm` package; it's a thin consumer of it, same as `scripts/ask.py`. |
| `pdf_md.py` | One-off utility: converts the official ICC Laws PDF (`ilovepdf_merged.pdf`) into `data/icc_rulebook.md` via `docling`. Not part of the ingestion pipeline itself. |

---

## `tests/` — what's covered

| File | Covers |
|---|---|
| `test_hashing.py` | SHA256 determinism for text and files. |
| `test_block_parser.py` | Code/table/list/curl segments detected as atomic and kept intact. |
| `test_markdown_structure.py` | Header hierarchy → metadata, no content leaking across sections. |
| `test_semantic_chunker.py` | Code/curl/table blocks never split at normal sizes; oversized blocks get force-split with `forced_split=True`; chunk metadata (hash, header_path) present and unique. |
| `test_embeddings.py` | Per-batch caching survives a later batch's failure (resume works); a retried call only re-embeds what's missing; cache doesn't leak between `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` for identical text. |
| `test_vectorstore.py` | Index gets created (with cosine metric) if missing, not recreated if it already exists; upsert packs chunk content into Pinecone metadata; query unpacks it back out and converts similarity → distance; `existing_ids`/metadata sanitization behave correctly. All against a fake in-memory Pinecone client double, no real account needed. |

Run everything:
```bash
pip install -r requirements.txt
pytest -q
```
No live API key needed — all tests either avoid real embedding calls
(`use_semantic_chunking=False`, `embeddings=None`) or fake out
`_embed_with_retry` directly.

Lint/type-check:
```bash
pip install ruff mypy
ruff check src tests scripts
mypy --ignore-missing-imports src/cricllm
```

---

## Incident playbook (things that have actually gone wrong)

**`404 NOT_FOUND ... models/text-embedding-004 is not found`**
Google retired that model. Fixed — `config.py` defaults to
`models/gemini-embedding-001`. If this recurs with a *different* model name,
list what your key currently has access to:
```python
from google import genai
client = genai.Client(api_key="...")
for m in client.models.list():
    if 'embedContent' in (m.supported_actions or []):
        print(m.name)
```

**`429 ... embed_content_free_tier_requests, limit: 100`**
Free-tier cap: 100 `embedContent` requests/minute, per user/project/model.
Not a bug. `embeddings.py` retries with backoff automatically. Avoid manually
re-running the script back-to-back — each restart plus its own retries adds
more requests to the same rolling window, extending the outage. Wait ~90s
untouched, run once, let it finish.

**A batch fails partway through a large ingestion run**
Before the per-batch caching fix, this discarded all progress from earlier
batches in the same call, forcing full re-embeds on every retry. Fixed —
`embeddings.py::_embed_in_batches` now caches each batch the instant it
succeeds. A re-run after a partial failure only re-embeds the genuine
remainder.

**Warnings like `code segment of 6604 tokens exceeds the embedding API's input limit`**
Expected and handled — `semantic_chunker.py` force-splits anything above
`CRICLLM_MAX_EMBEDDING_INPUT_TOKENS` (2048) since Gemini would otherwise
reject it outright. Segments between the "hard max" (1400) and 2048 are
kept whole on purpose (fine to embed, just bigger than the target).

**`TypeError: get_many() takes 3 positional arguments but 4 were given`**
Version skew — `embeddings.py` was updated to pass `task_type` to the cache,
but `cache.py` on that machine still has the old 2-arg signature. Re-copy
the whole project directory rather than patching individual files.

**`ask.py` says "I do not have enough information" even though sources look relevant**
Check the actual distances in `--- Sources ---`. Low distances (~0.3-0.5)
with a "not enough info" answer usually means the retrieved section is
topically related but doesn't state the specific fact asked (verify by
uncommenting `print(match["content"])` in `ask.py` to see the raw retrieved
text). High distances mean the vector store likely doesn't have good
coverage for that topic yet — check `store.count()` and whether ingestion
ever completed without errors.

**Vector store has 0 chunks / `ask.py` tells you to run ingestion first**
Ingestion never completed successfully for that collection — check
`logs/dead_letter/` for failed-batch records and `logs/cricllm.log` for the
actual failure reason (usually one of the above).

**Deployed on Render free tier: repeated `WORKER TIMEOUT` / `SIGKILL`, no obvious slow request**
Free tier gives ~0.1 CPU — under that much throttling, gunicorn's worker
can fail to "check in" with its own master process in time even while
sitting idle, since it never gets scheduled enough CPU to run its own
heartbeat code, and gets killed as if it were unresponsive. Free tier also
has no persistent disk and spins down after inactivity, which doesn't mix
well with anything needing local state. If you're seeing timeouts with no
correlated slow request in the logs, check the plan/tier before debugging
the app further — this turned out to be a real, separate issue from the
chromadb one below, not a symptom of it.

**chromadb hangs indefinitely inside a live gunicorn worker (the reason we're on Pinecone now)**
This was the big one. Deployed on Render, `/api/ask` would hang forever —
no error, no timeout, no response — until gunicorn's own `--timeout`
killed the worker (`WORKER TIMEOUT` / `SIGKILL`, "Perhaps out of memory?").
That message is misleading boilerplate; here's what we actually ruled out,
in order, with hard evidence each time:

- **Not the vector store data** — a direct check from the same shell showed
  the correct chunk count, matching local.
- **Not the deployed code** — confirmed the live commit had the right
  source (a stale `pip install -e git+...@<old commit>` in `requirements.txt`
  was a real but separate bug — it shadows the repo's actual `src/` code
  with a frozen old copy unless something explicitly puts `src/` first on
  `sys.path`, which `app.py`/`scripts/*.py` do; remove that `-e git+...`
  line if you ever see it, it should never be there).
- **Not the API key** — a bounded `curl` straight to Gemini's REST endpoint,
  and the full `QAEngine.answer_stream()` loop run directly in Python (not
  through Flask), both completed in ~10s.
- **Not gunicorn's worker class** — switching `sync` → `gthread` made zero
  difference; the hang happened at the identical line either way, which
  rules out anything signal-based (`sync`'s timeout mechanism).
- **Not memory** — sampled the worker's `VmRSS` every 10s during the hang:
  completely flat the whole time. A real leak/OOM climbs; this didn't move
  at all. (Kernel OOM kills also show up in `/sys/fs/cgroup/memory.events`'s
  `oom_kill` counter — worth checking directly if you ever suspect this
  again, rather than trusting gunicorn's canned message.)
- **Not a stale/reused connection** — a *brand-new* `VectorStore`,
  constructed fresh inside the request, hung identically to the long-lived
  one.
- **Not Render's reverse proxy** — hitting `127.0.0.1:<port>` directly, from
  inside the same container, hung exactly the same way.

What was left: any chromadb call — construction or a plain `.count()` —
worked fine once, at module import time when the worker first boots (that's
the only reason `GET /` ever succeeded), but hung the instant it was called
again from inside actual request-handling code, in every configuration
tried. A raw `sqlite3` query against the identical `.chroma` file, from
inside that same live request, was instant. So it wasn't SQLite, wasn't the
file, wasn't our code — it was something specific to chromadb's own
(Rust/PyO3-backed) client behaving differently once a gunicorn worker is
actively serving versus still starting up. We didn't get further than that
diagnosis (no `py-spy`/`sudo` available in the container to get an actual
stack trace of the stuck frame) — instead of chasing it further, we moved
the vector store off local chromadb entirely onto hosted Pinecone (a plain
REST/HTTP client, no local persistent Rust core), which sidesteps the
question rather than answering it.

If you're ever debugging something that looks like this again — hangs only
in a live server process, not standalone; not memory; not the obvious
suspects — the technique that actually worked was **bisection via
temporary debug routes**: add a route that does the minimal suspect
operation directly (bypass every layer above it) and log/print at each
step, redeploy, reproduce, read the logs. That's what actually narrowed
this down through six ruled-out hypotheses without ever getting a real
stack trace.
