# AI_Academy_SWE_FinalProject -- Topic 1: Lost and Found
This repository is dedicated for final project of Software Engineering module in AI Academy.

Structure of the repository:
team-lostfound/
│
├── README.md
├── .env.example
├── .gitignore
├── Dockerfile
├── Topic
├── requirements-ai.txt
├── requirements.txt
│
├── ai/                        ← copied from topic-1, never edited
│   ├── __init__.py
│   ├── embedding.py
│   ├── schemas.py
│   ├── similarity.py
│   ├── vlm.py
│   └── providers/
│
├── src/                       ← everything we build
│   ├── config.py
│   ├── models.py
│   ├── cli.py
│   ├── api.py
│   ├── services/
│   │   └── ai_service.py
│   ├── storage/
│   │   └── repository.py
│   └── concurrency/
│       └── pipeline.py
│
├── tests/                     ← given smoke tests + our tests
│   ├── conftest.py            ← from provided folder
│   ├── test_ai_smoke.py       ← from provided folder, never weaken
│   ├── test_services.py       ← we write
│   ├── test_concurrency.py    ← we write
│   └── test_end_to_end.py     ← we write
│
├── data/                      ← copied from topic-1
│   ├── _make_samples.py
│   ├── lost/
│   └── found/
│
├── scripts/
│   └── demo.py                ← the graded demo scenario
│
├── artefacts/
│   └── demo_output.txt        ← saved output of one real run
│
└── report/
    ├── report.tex
    └── report.pdf
