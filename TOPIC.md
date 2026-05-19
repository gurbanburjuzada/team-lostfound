# Topic 1 — Smart Lost & Found

> **What you receive:** a working AI module (VLM + embedding + similarity), sample images, an end-to-end demo, and smoke tests.
> **What you build:** the full software-engineering layer around it (storage, HTTP API, CLI, concurrency, retries, logging, validation, tests, Docker, README, report).

---

## The problem

Users register **lost** items (image + free-text description) and **found** items (image + free-text description). The service computes the most likely matches between the two pools using a vision-language description plus an embedding similarity score over those descriptions.

## What the AI does

1. **VLM** (`ai.describe_item`) takes an image + the user's text, asks the chosen vision-language model to produce a structured description (object class, colours, brand, distinguishing marks, location hints, confidence). The output is a typed `ItemDescription` pydantic model.
2. **Embedding** (`ai.embed`) takes the flattened description text and returns a unit-normalized vector from the chosen embedding provider.
3. **Similarity** (`ai.cosine`, `ai.top_k`) is pure NumPy: cosine similarity and a top-k helper.

Everything is provider-agnostic. Switch providers via environment variables (no code change):

```bash
LLM_PROVIDER=anthropic         # or openai, gemini
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...

EMBEDDING_PROVIDER=openai      # or gemini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=...
```

(Anthropic does not currently offer a first-party embedding endpoint — pair Claude with an OpenAI or Gemini embedder.)

## What you build (the SE layer)

| Component | Required | Notes |
|---|---|---|
| `config.py` | yes | Read env, expose typed settings (`pydantic-settings` recommended). |
| Storage | yes | PostgreSQL (via `psycopg` or `asyncpg`) for metadata; filesystem for image blobs. |
| HTTP API | **yes** | `POST /items/lost`, `POST /items/found`, `GET /items/{id}/matches?k=N`, `GET /items?status=...`. FastAPI or Flask. |
| CLI | yes | `register-lost`, `register-found`, `search-matches`, `list`. |
| Concurrency | yes | When registering a batch, embedding + scoring run via `asyncio.gather`. |
| Retries | yes | Exponential backoff on every `ai.*` call. `tenacity` recommended. |
| Caching | yes | Re-embedding the same description within a session must hit a cache. |
| Validation | yes | Reject non-JPEG/PNG, oversize files, missing fields. |
| Logging | yes | `logging` module, env-driven level. |
| Tests | yes | ≥60% coverage, all offline. |
| Dockerfile | yes | Builds and runs end-to-end. |
| README | yes | Setup, env, run, test, curl examples. |

## How to run what we shipped

```bash
# (1) Install deps used by the AI layer:
pip install numpy pydantic

# Optional, only needed if you actually call the providers:
pip install anthropic openai google-genai

# (2) Generate the sample images (one-time):
python data/_make_samples.py

# (3) Try the offline demo (no API keys required, no network):
python demo_ai.py --offline

# (4) Run the smoke tests (offline, no network):
pytest tests/test_ai_smoke.py -v
```

## The contract (do not break)

- **Do not** edit any file under `ai/`. If you find a bug, file an issue with the instructor.
- **Do not** delete or weaken `tests/test_ai_smoke.py`. These tests are run during grading; they must pass on your final repo.
- **Do not** call provider SDKs directly from your business logic. Always go through `ai.describe_item`, `ai.embed`, and the `ai.providers` factory.

## Recommended folder layout for your project

```
your-project/
├── ai/                       # COPIED FROM HERE, unchanged
├── src/
│   ├── config.py
│   ├── models.py             # YOUR pydantic models: Item, MatchRecord, ...
│   ├── services/
│   │   ├── ai_service.py     # retries, caching, logging around ai.*
│   │   └── ...
│   ├── core/
│   │   └── matcher.py        # business logic
│   ├── concurrency/
│   │   └── pipeline.py       # asyncio orchestration
│   ├── storage/
│   │   └── repository.py     # PostgreSQL + filesystem
│   ├── cli.py
│   └── api.py                # FastAPI / Flask
├── tests/                    # YOUR tests + the provided smoke tests
├── data/                     # COPIED FROM HERE
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Sample data

`data/lost/` and `data/found/` contain 5 + 7 small synthetic PNGs. The filenames are descriptive so the offline `_OfflineVLM` in `demo_ai.py` can derive a plausible description. Replace these with real photos for development; keep them small for grading.

## Free-tier API options

| Provider | Free tier? | Notes |
|---|---|---|
| Anthropic Claude | Limited trial credit | Best for VLM quality. Pair with OpenAI embeddings. |
| OpenAI GPT-4o-mini | Pay-as-you-go (cheap) | Smallest cost per VLM call. |
| Google Gemini | Generous free tier | Both VLM and embeddings. |

A single demo run on the 12 sample images is well under $0.05 on any of these.
