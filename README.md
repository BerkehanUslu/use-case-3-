# Immobilier Intelligent — Assistant Immobilier Français

An AI-powered real estate assistant for the French market built with FastAPI, Streamlit, HuggingFace Transformers, and FAISS.

---

## Project Description

This project implements an intelligent assistant for French real estate, covering:

- **DVF data aggregation** — processes raw government transaction data (Demandes de Valeurs Foncières) into per-commune price statistics
- **Semantic property search** — FAISS vector index over commune descriptions (sentence-transformers)
- **Q&A on property listings** — extractive question answering with CamemBERT (etalab-ia/camembert-base-squadFR-fquad-piaf)
- **Text summarization** — abstractive summarization of property reports with BARThez
- **Sentiment analysis** — BERT multilingual sentiment scoring of property descriptions
- **REST API** — FastAPI service exposing all models
- **Interactive dashboard** — Streamlit app with 5 analysis tabs

---

## Architecture

```
Raw DVF files (*.txt / *.csv)
        │
        ▼
01_data_preparation.py  ──►  data/dvf_communes.parquet
        │
        ▼
02_vector_indexing.py   ──►  indexes/real_estate.faiss
                              indexes/real_estate_meta.json
        │
        ├── 03_qa_system.py        (CamemBERT extractive QA)
        ├── 04_summarization.py    (BARThez abstractive summarization)
        └── 05_sentiment_analysis.py (BERT multilingual sentiment)
                │
                ▼
        06_api_deployment.py  ──►  FastAPI on :8000
                │
                ▼
        07_streamlit_app.py   ──►  Streamlit UI on :8501
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) Docker + Docker Compose

### Local setup

```bash
cd backend/

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Prepare data

Place raw DVF pipe-delimited `.txt` files in `backend/data/`, then:

```bash
python 01_data_preparation.py
# Output: data/dvf_communes.parquet
```

### 2. Build the vector index

```bash
python 02_vector_indexing.py
# Output: indexes/real_estate.faiss + indexes/real_estate_meta.json
```

### 3. Start the API

```bash
uvicorn 06_api_deployment:app --reload --host 0.0.0.0 --port 8000
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 4. Start the Streamlit dashboard

```bash
streamlit run 07_streamlit_app.py --server.port 8501
# Dashboard available at http://localhost:8501
```

---

## Docker Deployment

Both services (API + Streamlit) are defined in `backend/docker-compose.yml`.

```bash
cd backend/
docker compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI API | http://localhost:8000 |
| Streamlit dashboard | http://localhost:8501 |

```bash
docker compose down   # Stop all services
```

---

## Repository Structure

```
backend/
├── 01_data_preparation.py     # DVF aggregation pipeline
├── 02_vector_indexing.py      # FAISS index builder
├── 03_qa_system.py            # CamemBERT Q&A
├── 04_summarization.py        # BARThez summarization
├── 05_sentiment_analysis.py   # Sentiment analysis
├── 06_api_deployment.py       # FastAPI REST API
├── 07_streamlit_app.py        # Streamlit UI (5 tabs)
├── config.py                  # Centralised configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                      # DVF source files + generated Parquet
└── indexes/                   # Generated FAISS index files
```

---

## Data Sources

- **DVF** — [Demandes de Valeurs Foncières](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) (Ministère de l'Économie / DGFiP)
- **BAN** — [Base Adresse Nationale](https://adresse.data.gouv.fr/data/ban/adresses/latest/) (Etalab / La Poste)

Data published under [Licence Ouverte / Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence).
