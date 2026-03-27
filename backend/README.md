# Immobilier Intelligent — Backend

Python backend for the French real-estate intelligent assistant.
Built with FastAPI, Streamlit, HuggingFace Transformers, FAISS, and the
open French government datasets DVF and BAN.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Architecture](#architecture)
4. [Setup](#setup)
5. [Running Each Script](#running-each-script)
6. [API Endpoints](#api-endpoints)
7. [Testing](#testing)
8. [Docker Deployment](#docker-deployment)
9. [Environment Variables](#environment-variables)

---

## Project Overview

This backend provides:

| Component | Description |
|-----------|-------------|
| Data pipeline | DVF aggregation by commune / year / property type |
| NLP models | QA (CamemBERT), summarisation (BARThez), sentiment, embeddings |
| Vector search | FAISS index for semantic property search |
| REST API | FastAPI service exposing all models and data |
| Dashboard | Streamlit app for interactive exploration |

Current validated status:
- Full pipeline tests pass locally: `36 PASS / 0 FAIL / 2 SKIP`
- Response-time target validated in tests: all checked endpoints under 2 seconds
- Q&A benchmark validated in full mode: `7/7 = 100%`
- Docker build and startup validated for FastAPI and Streamlit
- Remaining environment-dependent caveat: BAN geocoding may fail when network access to BAN is unavailable, which leaves the interactive map in mock mode

---

## Data Sources

### DVF — Demandes de Valeurs Foncières
- **Publisher:** Ministère de l'Économie / DGFiP
- **URL:** <https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/>
- **Format:** CSV (one file per year, ~multi-GB)
- **Key fields:** `valeur_fonciere`, `surface_reelle_bati`, `type_local`,
  `code_commune`, `nom_commune`, `date_mutation`

### BAN — Base Adresse Nationale
- **Publisher:** Etalab / La Poste
- **URL:** <https://adresse.data.gouv.fr/data/ban/adresses/latest/>
- **Format:** CSV GZ by department
- **Key fields:** `lon`, `lat`, `numero`, `nom_voie`, `code_postal`,
  `nom_commune`, `code_insee`

---

## Architecture

```
backend/
├── config.py                  # Centralised configuration
├── 01_data_preparation.py     # DVF aggregation pipeline
├── 02_vector_indexing.py      # FAISS index builder
├── 03_qa_system.py            # CamemBERT extractive QA
├── 04_summarization.py        # BARThez summarization
├── 05_sentiment_analysis.py   # BERT multilingual sentiment
├── 06_api_deployment.py       # FastAPI REST API
├── 07_streamlit_app.py        # Streamlit dashboard (5 tabs)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                      # DVF source files + generated Parquet
│   ├── dvf_communes.parquet   # (generated)
│   └── *.txt / *.csv          # Raw DVF files (place here before running)
└── indexes/                   # Generated FAISS index files
    ├── real_estate.faiss
    └── real_estate_meta.json
```

---

## Setup

### Prerequisites

- Python 3.11+
- `pip` or `conda`
- Docker + Docker Compose (for containerised deployment)

### Local Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download real DVF data
#    Place the raw CSV at data/raw_dvf.csv before running the pipeline.
```

---

## Running Each Script

All scripts must be executed from the `backend/` directory.

```bash
cd backend/
```

### 01 — Data Preparation

Aggregates DVF transaction data. Falls back to realistic mock data when no
real DVF file is available.

```bash
python 01_data_preparation.py
```

Output:
- `data/dvf_communes.parquet` — aggregated by commune / year / property type
- `data/sample_dvf.csv` — 50-row mock subset for quick testing

### 02 — Build FAISS Vector Index

Builds sentence-transformer embeddings over commune descriptions and stores them in a FAISS index.

```bash
python 02_vector_indexing.py
```

If the environment cannot reach Hugging Face or BAN, the rebuild may fail or produce metadata without coordinates. In that case:
- semantic search keeps working with the last valid FAISS index already present in `indexes/`
- the interactive map falls back to mock data until BAN coordinates are successfully generated

Output:
- `indexes/real_estate.faiss` — binary FAISS index
- `indexes/real_estate_meta.json` — metadata (commune names, descriptions, price data)

### 03 — Q&A System

Run standalone Q&A against a property description using CamemBERT.

```bash
python 03_qa_system.py
```

### 04 — Summarization

Run standalone summarization demo using BARThez.

```bash
python 04_summarization.py
```

### 05 — Sentiment Analysis

Run standalone sentiment analysis demo using multilingual BERT.

```bash
python 05_sentiment_analysis.py
```

### 06 — FastAPI REST API

```bash
uvicorn 06_api_deployment:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs available at <http://localhost:8000/docs>.

### 07 — Streamlit Dashboard

```bash
streamlit run 07_streamlit_app.py --server.port 8501
```

The dashboard reads the API URL from:
- `API_URL`
- or `API_HOST` + `API_PORT`
- default fallback: `http://localhost:8000`

---

## API Endpoints

> The API is implemented in `06_api_deployment.py`. Start it with `uvicorn 06_api_deployment:app --reload`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/communes` | List all communes with price data |
| GET | `/communes/{commune}` | Aggregate price metrics for a commune |
| GET | `/trends/{commune}` | Year-over-year trend data for a commune |
| POST | `/search` | Semantic property search (FAISS) |
| POST | `/qa` | Question answering on a property description |
| POST | `/summarize` | Summarise a property listing |
| POST | `/summarize-pdf` | Summarise an uploaded PDF report |
| POST | `/sentiment` | Sentiment analysis on review text |

### Example — Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Example — Commune Query

```bash
curl http://localhost:8000/communes/Paris
```

### Example — QA

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"context": "Appartement de 65m² au 3ème étage ...", "question": "Quelle est la surface ?"}'
```

---

## Testing

Run the automated pipeline checks from `backend/`:

```bash
python test_pipeline.py
python test_pipeline.py --full
```

Validated results on the current project state:
- Fast mode: mock/integration checks for data prep, API, Streamlit hooks, response times
- Full mode: real-model validation for Q&A, summarization, and sentiment
- Latest validated full result: `36 PASS / 0 FAIL / 2 SKIP`

The two remaining skips are BAN-related:
- live BAN endpoint availability
- coordinate population in generated metadata when BAN is unreachable

---

## Docker Deployment

### Build and Start All Services

```bash
docker compose up --build
```

This starts:

| Service | URL |
|---------|-----|
| FastAPI API | <http://localhost:8000> |
| Streamlit dashboard | <http://localhost:8501> |

Verified working:
- FastAPI container serves `/health` and `/docs`
- Streamlit container starts successfully and connects to the API via `http://api:8000`

### Start a Single Service

```bash
docker compose up api         # API only
docker compose up streamlit   # Dashboard only
```

### Stop All Services

```bash
docker compose down
```

### View Logs

```bash
docker compose logs -f api
docker compose logs -f streamlit
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | unset | Full API base URL for Streamlit, e.g. `http://api:8000` in Docker |
| `API_HOST` | `0.0.0.0` | Host the FastAPI server binds to |
| `API_PORT` | `8000` | Port the FastAPI server listens on |

Variables can be set in a `.env` file in the `backend/` directory or passed
directly to `docker compose`.

```bash
# .env (do not commit to git)
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run `python 01_data_preparation.py` to validate the data pipeline.
3. Add tests where applicable.
4. Open a pull request against `main`.

---

## License

This project uses open government data published under the
[Licence Ouverte / Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence).
