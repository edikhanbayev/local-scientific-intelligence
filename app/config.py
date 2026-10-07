import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
METADATA_FILE = DATA_DIR / "metadata.csv"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:0.6b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "cointegrated/rubert-tiny2"
)

COLLECTION_NAME = "scientific_documents"