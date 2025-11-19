"""Configuration management for Axis Guidelines AI"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
PDF_FOLDER = BASE_DIR / 'test_pdfs'
METADATA_FILE = BASE_DIR / 'pdf_metadata.json'
CHUNKS_FILE = BASE_DIR / 'chunks.pkl'
INDEX_FILE = BASE_DIR / 'index.faiss'
EMBEDDINGS_FILE = BASE_DIR / 'embeddings.npy'
LOG_FILE = BASE_DIR / 'processor.log'

# Processing parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.95
MIN_CHUNK_LENGTH = 50

# Model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GEMINI_MODEL = 'gemini-2.5-flash-lite'
GEMINI_TEMPERATURE = 0

# Performance settings
MAX_CACHE_SIZE = 100
MAX_HISTORY_SIZE = 10
SEARCH_TOP_K = 3
QUERY_TIMEOUT = 30  # seconds
BATCH_SIZE = 32  # for embedding generation

# API settings
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Create directories if they don't exist
PDF_FOLDER.mkdir(exist_ok=True)
