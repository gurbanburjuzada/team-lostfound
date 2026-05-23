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

### Using Docker Compose (recommended for development)

The easiest way to run the full stack (app + PostgreSQL) locally:

```bash
# Start both the API and PostgreSQL database
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop everything
docker-compose down
```

Once running:
- **API**: http://localhost:8000
- **Health check**: http://localhost:8000/health
- **PostgreSQL**: localhost:5432 (user: `postgres`, password: `postgres`, db: `lostfound`)

**Configuration**: Environment variables are in `docker-compose.yml`. To override, create a `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### Manual Docker build and run

If you prefer to use Docker directly without compose:

```bash
# Build the image
docker build -t lostfound:latest .

# Run the image (requires external PostgreSQL)
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  lostfound:latest
```

**Note**: PostgreSQL must be reachable from inside the container. If running
PostgreSQL separately, update `DATABASE_URL` in `.env` to point to your host IP
(e.g., `postgresql+asyncpg://postgres:dev@host.docker.internal:5432/lostfound`).

### Run the demo in Docker

With docker-compose already running:

```bash
docker-compose exec app python scripts/demo.py
```

Or standalone (requires all env vars + external PostgreSQL):

```bash
docker run --rm \
  --env-file .env \
  --network host \
  lostfound:latest \
  python scripts/demo.py
```

With docker-compose already running:

```bash
docker-compose exec app python scripts/demo.py
```

Or standalone (requires all env vars + external PostgreSQL):

```bash
docker run --rm \
  --env-file .env \
  --network host \
  lostfound:latest \
  python scripts/demo.py
```

---

## Troubleshooting

### Database connection errors

**Error**: `psycopg.OperationalError: could not connect to server`

**Solution**:
1. Verify PostgreSQL is running: `docker ps | grep postgres`
2. Check `DATABASE_URL` in `.env` — it should match your PostgreSQL host/port
3. For docker-compose, use `postgres` as the hostname (not `localhost`)
4. For Docker on Mac/Windows, use `host.docker.internal` as the hostname

### Missing API keys

**Error**: `ProviderError: API key not set`

**Solution**:
1. Create `.env` from `.env.example`: `cp .env.example .env`
2. Fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and/or `GOOGLE_API_KEY`
3. Restart the app: `docker-compose restart app`

### "Item is still being processed"

**Error**: `409 Conflict — Item is still being processed`

**Cause**: The embedding for the item hasn't been computed yet (still in `register_batch`).

**Solution**: Wait a few seconds and retry the query.

### "No matches found"

This is normal if:
- No items of the opposite status exist (e.g., querying a LOST item when no FOUND items are registered)
- All similarity scores are below the threshold
- Try registering more sample items: `python data/_make_samples.py && python scripts/demo.py`

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