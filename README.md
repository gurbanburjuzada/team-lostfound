# Smart Lost & Found — AI Academy SWE Final Project

Topic 1 of the AI-ENG-110 Software Engineering final project.
Users register **lost** and **found** items (image + description). The system uses
a vision-language model to describe each item and embedding similarity to surface
the most likely matches across the two pools.

---

## Team

| Member | Branch prefix | Main responsibility |
|---|---|---|
| Gurban | `gurban/` | Config, models, repo setup |
| Murad | `murad/` | AI service, storage, pipeline, core matcher |
| Davud | `davud/` | HTTP API, CLI, Dockerfile, tests, demo |

---

## Project structure
team-lostfound/
├── ai/                  ← provided, do not edit
├── src/
│   ├── config.py        ← pydantic-settings, all env vars
│   ├── models.py        ← SE-layer Pydantic schemas
│   ├── api.py           ← FastAPI HTTP server
│   ├── cli.py           ← Click CLI
│   ├── services/
│   │   └── ai_service.py   ← retries, cache, logging around ai.*
│   ├── core/
│   │   └── matcher.py      ← top-k matching business logic
│   ├── concurrency/
│   │   └── pipeline.py     ← asyncio batch registration
│   └── storage/
│       └── repository.py   ← SQLAlchemy ORM (PostgreSQL) + filesystem
├── tests/
├── scripts/
│   └── demo.py
├── data/
├── artefacts/
├── Dockerfile
├── requirements.txt
└── .env.example

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<org>/team-lostfound.git
cd team-lostfound
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | yes | `anthropic`, `openai`, or `gemini` |
| `LLM_MODEL` | yes | e.g. `claude-sonnet-4-6` |
| `ANTHROPIC_API_KEY` | if anthropic | Your Anthropic key |
| `OPENAI_API_KEY` | if openai/embeddings | Your OpenAI key |
| `GOOGLE_API_KEY` | if gemini | Your Google key |
| `EMBEDDING_PROVIDER` | yes | `openai` or `gemini` |
| `EMBEDDING_MODEL` | yes | e.g. `text-embedding-3-small` |
| `DATABASE_URL` | yes | `postgresql+asyncpg://user:pass@host:5432/lostfound` |
| `IMAGE_STORAGE_DIR` | yes | Path to store uploaded images (e.g. `./storage/images`) |
| `MAX_IMAGE_SIZE_MB` | no | Default `5` |
| `LOG_LEVEL` | no | Default `INFO` |
| `API_HOST` | no | Default `0.0.0.0` |
| `API_PORT` | no | Default `8000` |
| `SEMAPHORE_LIMIT` | no | Max parallel AI calls, default `5` |
| `CACHE_TTL_SECONDS` | no | Embedding cache TTL, default `3600` |
| `RETRY_MAX_ATTEMPTS` | no | Default `3` |
| `RETRY_WAIT_SECONDS` | no | Default `1.0` |

### 3. Start PostgreSQL

```bash
docker run -d \
  --name lostfound-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=lostfound \
  -p 5432:5432 \
  postgres:16
```

The app creates the `items` table automatically on first start.

### 4. Generate sample images (one-time)

```bash
python data/_make_samples.py
```

---

## Running

### Offline smoke test (no API keys needed)

```bash
pytest tests/test_ai_smoke.py -v
```

### CLI

```bash
# Register a lost item
python -m src.cli register-lost data/lost/umbrella_black.png "black umbrella left at library"

# Register a found item
python -m src.cli register-found data/found/umbrella_black_2.png "found a dark umbrella near entrance"

# Search top-3 matches for a lost item
python -m src.cli search-matches <item-id> --k 3

# List all items (or filter by status)
python -m src.cli list
python -m src.cli list --status lost
```

### HTTP API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

#### curl examples

```bash
# Register a lost item
curl -X POST http://localhost:8000/items/lost \
  -F "description=black umbrella left at library" \
  -F "image=@data/lost/umbrella_black.png"

# Register a found item
curl -X POST http://localhost:8000/items/found \
  -F "description=found a dark umbrella near entrance" \
  -F "image=@data/found/umbrella_black_2.png"

# Get top-3 matches for a lost item
curl "http://localhost:8000/items/<item-id>/matches?k=3"

# List all found items
curl "http://localhost:8000/items?status=found"
```

### Graded demo script

```bash
python scripts/demo.py
```

Registers all items in `data/lost/` and `data/found/`, then queries one lost item
and prints the top-3 matches with similarity scores. Output is saved to
`artefacts/demo_output.txt`.

---

## Testing

```bash
# Run all tests (offline, no network)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run only the provided AI smoke tests
pytest tests/test_ai_smoke.py -v
```

Coverage target: **≥ 60%**.

---

## Concurrency benchmark

Sequential vs concurrent batch registration of all 12 sample images:

```bash
# Sequential
python scripts/demo.py --mode sequential

# Concurrent (asyncio.gather, semaphore=5)
python scripts/demo.py --mode concurrent
```

| Mode | Items | Wall-clock time |
|---|---|---|
| Sequential | 12 | _fill after run_ |
| Concurrent | 12 | _fill after run_ |

Bottleneck: VLM + embedding calls (I/O-bound). Bounded by `SEMAPHORE_LIMIT` to
respect provider rate limits.

---

## Docker

### Build

```bash
docker build -t lostfound:latest .
```

### Run

```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  lostfound:latest
```

### Full demo in Docker

```bash
docker run --rm \
  --env-file .env \
  --network host \
  lostfound:latest \
  python scripts/demo.py
```

> PostgreSQL must be reachable from inside the container. Update `DATABASE_URL`
> in `.env` to point to your host IP (not `localhost`) when running in Docker.

---

## Architecture

See `docs/architecture.md` for the full diagram.

CLI / HTTP API
│
▼
AIService (retries, cache, logging)
│
├── ai.describe_item (VLM)
└── ai.embed (embedding)
│
▼
Matcher (top-k cosine similarity)
│
▼
Repository ──► PostgreSQL (metadata, ORM)
└──► Filesystem (image blobs)

---

## AI disclosure

Portions of this codebase were drafted with the assistance of AI coding tools
(Claude). All code has been reviewed, understood, and is defensible by the team
in the oral examination.