import os

DATA_DIR = "data/"
INDEX_DIR = "indexes/"
DVF_PARQUET = "data/dvf_communes.parquet"
DVF_SAMPLE = "data/sample_dvf.csv"
FAISS_INDEX_PATH = "indexes/real_estate.faiss"
FAISS_META_PATH = "indexes/real_estate_meta.json"
MODELS = {
    "qa": "etalab-ia/camembert-base-squadFR-fquad-piaf",
    "summarizer": "moussaKam/barthez-orangesum-abstract",
    "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
    "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
