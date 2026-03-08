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
7. [Docker Deployment](#docker-deployment)
8. [Environment Variables](#environment-variables)

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
├── 02_api.py                  # FastAPI application          (coming soon)
├── 03_embeddings.py           # FAISS index builder          (coming soon)
├── 04_qa.py                   # QA model wrapper             (coming soon)
├── 05_summarizer.py           # Summarisation model          (coming soon)
├── 06_sentiment.py            # Sentiment analysis           (coming soon)
├── 07_streamlit_app.py        # Streamlit dashboard          (coming soon)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                      # Generated at runtime
│   ├── dvf_communes.parquet
│   └── sample_dvf.csv
└── indexes/                   # Generated at runtime
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

### 02 — API (coming soon)

```bash
uvicorn 02_api:app --reload --host 0.0.0.0 --port 8000
```

### 03 — Build FAISS Index (coming soon)

```bash
python 03_embeddings.py
```

### 07 — Streamlit Dashboard (coming soon)

```bash
streamlit run 07_streamlit_app.py --server.port 8501
```

---

## API Endpoints

> The API is implemented in `02_api.py` (coming soon).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/communes` | List all communes with price data |
| GET | `/prix/{code_commune}` | Price history for a commune |
| POST | `/search` | Semantic property search (FAISS) |
| POST | `/qa` | Question answering on a property description |
| POST | `/summarize` | Summarise a property listing |
| POST | `/sentiment` | Sentiment analysis on review text |

### Example — Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Example — Price Query

```bash
curl http://localhost:8000/prix/75056
```

### Example — QA

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"context": "Appartement de 65m² au 3ème étage ...", "question": "Quelle est la surface ?"}'
```

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
