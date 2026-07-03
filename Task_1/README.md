# Dynamic Knowledge Base System

A production-ready system that keeps a chatbot's vector database continuously
up to date with new information pulled from external sources (files, web
pages, RSS feeds), on a per-source schedule — without re-embedding content
that hasn't changed.

## How it works

```
 ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
 │   Sources    │ -> │   Loaders    │ -> │  Chunk + Hash │ -> │   Manifest  │
 │ file/web/rss │    │ (pluggable)  │    │   documents   │    │  (sqlite)   │
 └─────────────┘    └──────────────┘    └───────┬───────┘    └──────┬──────┘
                                                  │  new/changed only │
                                                  v                   v
                                          ┌────────────────────────────────┐
                                          │      Vector Store (Chroma)      │
                                          │  persistent, embeds + upserts   │
                                          └────────────────┬─────────────────┘
                                                            │
                        ┌───────────────────────────────────┘
                        v
                ┌───────────────┐
                │    Chatbot     │  retrieves top-k chunks, optionally asks
                │  (retrieval +  │  Claude to compose a grounded answer
                │  optional LLM) │
                └───────────────┘

        A per-source APScheduler job re-runs this loop on its own cadence
        (interval or cron), so the knowledge base expands automatically.
```

Key design points:

- **Change detection, not blind re-indexing.** Every source document is
  content-hashed. Unchanged documents are skipped entirely on each run —
  only new or modified content gets re-chunked and re-embedded, which keeps
  update runs cheap even as the knowledge base grows.
- **Stale-content cleanup.** When a document's content changes, its old
  chunks are deleted before the new ones are inserted (no duplicate/outdated
  vectors lingering in the index). When a document disappears from its
  source entirely, its chunks are removed too.
- **Per-source scheduling.** A fast-moving RSS feed can refresh hourly while
  a static document folder rescans nightly — each source in `config.yaml`
  has its own `schedule` block.
- **Pluggable sources.** `file`, `web`, and `rss` loaders are included;
  adding a new source type (e.g. a database, Notion, Confluence, Slack) is a
  ~30-line class that returns a list of `RawDocument(doc_id, text, metadata)`.
- **No required external API.** Embeddings default to Chroma's bundled ONNX
  MiniLM model (small, fast, no GPU/torch needed). The chatbot answer step
  is retrieval-only by default and automatically upgrades to LLM-generated,
  context-grounded answers if you set `ANTHROPIC_API_KEY`.

## Project layout

```
kb_system/
├── config.yaml              # sources, schedules, chunking, embedding settings
├── main.py                  # CLI entrypoint (serve / update-now / chat / stats)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── sample_docs/             # example source content for the `file` loader
├── data/                    # persisted vector DB + manifest (created at runtime)
├── logs/                    # rotating-free plain log file (created at runtime)
└── src/
    ├── config.py            # YAML config loader
    ├── utils.py              # hashing, text chunking, logging setup
    ├── manifest.py            # sqlite change-tracking store
    ├── vector_store.py        # Chroma wrapper (upsert/delete/query)
    ├── updater.py             # orchestrates one update pass for a source
    ├── scheduler.py            # APScheduler wiring, one job per source
    ├── chatbot.py               # retrieval (+ optional Claude generation)
    └── loaders/
        ├── file_loader.py       # .txt / .md / .pdf from a directory
        ├── web_loader.py        # scrapes readable text from URLs
        └── rss_loader.py        # pulls entries from RSS/Atom feeds
```

## Quick start (local)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# optional: enables LLM-generated answers instead of raw passage dumps
cp .env.example .env && echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 1. Point config.yaml's `docs_folder` source at real content (or use sample_docs/ as-is)
# 2. Run a one-off update to populate the vector store
python main.py update-now

# 3. Ask it questions
python main.py chat

# 4. Check what's indexed
python main.py stats
```

## Running continuously (the "dynamic expansion" part)

```bash
python main.py serve
```

This starts the scheduler: each enabled source in `config.yaml` gets its own
job (interval or cron), the vector store is updated automatically in the
background, and the process stays alive until you send SIGINT/SIGTERM. No
restart, redeploy, or manual re-indexing is needed for new content to show up
in chatbot answers — it happens on the schedule you configured.

## Configuring sources

Edit `config.yaml`. Each source needs an `id`, `type`, `options`, and
`schedule`:

```yaml
sources:
  - id: "product_docs"
    type: "file"
    enabled: true
    options:
      path: "/data/product_docs"
      extensions: [".txt", ".md", ".pdf"]
      recursive: true
    schedule:
      type: "interval"
      minutes: 30

  - id: "changelog_feed"
    type: "rss"
    enabled: true
    options:
      feed_urls: ["https://example.com/changelog.xml"]
    schedule:
      type: "cron"
      hour: "*/4"        # every 4 hours
```

Schedule types:
- `interval`: `minutes` / `hours` / `seconds`
- `cron`: any [APScheduler CronTrigger](https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html) fields (`hour`, `minute`, `day_of_week`, ...)

## Adding a new source type

Create `src/loaders/my_loader.py`:

```python
class MyLoader:
    def __init__(self, source_id: str, options: dict):
        self.source_id = source_id
        self.options = options

    def load(self):
        from . import RawDocument
        # fetch your data, return a list of RawDocument(doc_id, text, metadata)
        return [RawDocument(doc_id="...", text="...", metadata={})]
```

Register it in `src/loaders/__init__.py`'s `LOADER_REGISTRY`, then reference
`type: "my_loader"` in `config.yaml`.

## Deployment (Docker)

```bash
docker compose up -d --build
```

- The vector DB and manifest are bind-mounted at `./data`, so they survive
  container restarts/redeploys.
- Edit `config.yaml` and `sample_docs/` (or mount your own content directory)
  and restart the container to pick up changes — the scheduler re-reads
  config on startup.
- Set `ANTHROPIC_API_KEY` in a `.env` file next to `docker-compose.yml` to
  enable LLM-generated answers.
- For a multi-source production setup, mount each content directory as its
  own volume and reference the corresponding paths in `config.yaml`.

## Embedding backend

By default, embeddings use Chroma's bundled ONNX MiniLM-L6-v2 model — no
torch dependency, small image, fast cold start, downloaded automatically on
first use (also pre-warmed at Docker build time). If you want higher-quality
embeddings and don't mind the extra ~2GB dependency, uncomment
`sentence-transformers` in `requirements.txt` and set
`vector_store.embedding_provider: "sentence_transformers"` in `config.yaml`.

## Testing

Core update logic (new-document embedding, skip-when-unchanged,
replace-on-change, delete-on-removal) is deterministic and can be verified
without any network access using a fake embedding function — see the pattern
in the module docstrings of `src/updater.py` for what's covered. In this
delivery it was verified end-to-end: new documents get embedded, unchanged
documents are skipped, changed documents have their stale chunks replaced,
and documents removed from a source have their vectors purged from the
store.

## Extending the chatbot

`src/chatbot.py`'s `KnowledgeBaseChatbot.ask()` is a small, self-contained
retrieval-augmented-generation loop — swap in a different LLM provider, add
re-ranking, or filter by `metadata` (e.g. restrict a query to one source)
using the `where` parameter already exposed on `VectorStore.query()`.
