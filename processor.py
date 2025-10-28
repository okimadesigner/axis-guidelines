import os
import hashlib
import logging
import re
from logging.handlers import RotatingFileHandler
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import json
from datetime import datetime
from tqdm import tqdm

# Import NLTK and download required data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from config import *

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class ChunkMetadata:
    """Metadata for each chunk"""
    def __init__(self, source_file, page_num, chunk_idx):
        self.source_file = source_file
        self.page_num = page_num
        self.chunk_idx = chunk_idx
        self.created_at = datetime.now().isoformat()

class IncrementalPDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Pre-compile regex patterns for efficiency
        self.text_cleaner = re.compile(
            r'<[^>]+>|[•◦○◆◇■□▪▫]|[^\x20-\x7E\n\s]'
        )
        self.whitespace_normalizer = re.compile(r'\s+')

        self.metadata = self.load_metadata()
        self.all_chunks = []
        self.chunk_sources = []  # Track which PDF each chunk came from

    def load_metadata(self):
        """Load metadata about previously processed PDFs"""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self):
        """Save metadata about processed PDFs"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_file_hash(self, filepath):
        """Calculate MD5 hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _create_chunks_from_text(self, text):
        """Create chunks from text with sentence boundary awareness"""
        # Clean text using compiled patterns
        text = self.text_cleaner.sub('', text)
        text = self.whitespace_normalizer.sub(' ', text)

        # Split into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding sentence exceeds chunk size, save current chunk
            if len(current_chunk) + len(sentence) > CHUNK_SIZE and current_chunk:
                if len(current_chunk) > MIN_CHUNK_LENGTH:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-20:] if len(words) > 20 else words
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
            else:
                current_chunk += ' ' + sentence

        # Add final chunk
        if len(current_chunk) > MIN_CHUNK_LENGTH:
            chunks.append(current_chunk.strip())

        return chunks

    def extract_chunks_from_pdf(self, pdf_path):
        """Extract text chunks from a single PDF with metadata"""
        reader = PdfReader(pdf_path)
        chunks_with_metadata = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            page_chunks = self._create_chunks_from_text(text)

            for chunk_idx, chunk in enumerate(page_chunks):
                chunks_with_metadata.append({
                    'text': chunk,
                    'metadata': ChunkMetadata(
                        source_file=os.path.basename(pdf_path),
                        page_num=page_num + 1,
                        chunk_idx=chunk_idx
                    ).__dict__
                })

        return chunks_with_metadata

    def compute_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def deduplicate_chunks(self, new_chunks, new_embeddings, existing_embeddings=None):
        """Remove duplicates using FAISS for efficient similarity search"""
        if len(new_chunks) == 0:
            return [], np.array([])

        if existing_embeddings is None or len(existing_embeddings) == 0:
            return new_chunks, new_embeddings

        # Use FAISS for efficient similarity search
        dimension = existing_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings
        normalized_existing = existing_embeddings.copy()
        faiss.normalize_L2(normalized_existing)
        temp_index.add(normalized_existing.astype('float32'))

        normalized_new = new_embeddings.copy()
        faiss.normalize_L2(normalized_new)

        # Batch search for duplicates
        D, I = temp_index.search(normalized_new.astype('float32'), k=1)

        unique_chunks = []
        unique_embeddings = []

        for i, (chunk, embedding, similarity) in enumerate(zip(new_chunks, new_embeddings, D[:, 0])):
            if similarity < SIMILARITY_THRESHOLD:
                unique_chunks.append(chunk)
                unique_embeddings.append(embedding)

        return unique_chunks, np.array(unique_embeddings) if unique_embeddings else np.array([])

    def process_pdfs(self):
        """Main processing function with incremental updates"""
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

        # Load existing data if available
        existing_chunks = []
        existing_embeddings = None
        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, 'rb') as f:
                existing_chunks = pickle.load(f)
            if os.path.exists(EMBEDDINGS_FILE):
                existing_embeddings = np.load(EMBEDDINGS_FILE)

        new_chunks = []
        files_to_process = []

        print("Checking for new or modified PDFs...")

        # Identify which PDFs need processing (first collect all files)
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            file_hash = self.get_file_hash(pdf_path)
            file_mtime = os.path.getmtime(pdf_path)

            # Check if file is new or modified
            if pdf_file not in self.metadata or self.metadata[pdf_file]['hash'] != file_hash:
                files_to_process.append(pdf_file)
                print(f"  - {pdf_file} (new or modified)")

        # Now process the files that need updating with progress bar
        for pdf_file in tqdm(files_to_process, desc="Processing PDFs"):
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)

            try:
                file_hash = self.get_file_hash(pdf_path)
                file_mtime = os.path.getmtime(pdf_path)

                logger.info(f"Processing: {pdf_file}")
                chunks_with_metadata = self.extract_chunks_from_pdf(pdf_path)

                # Extract just the text chunks for embedding
                text_chunks = [chunk['text'] for chunk in chunks_with_metadata]
                new_chunks.extend(text_chunks)

                self.metadata[pdf_file] = {
                    'hash': file_hash,
                    'modified': file_mtime,
                    'processed': datetime.now().isoformat(),
                    'chunk_count': len(chunks_with_metadata)
                }

            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}", exc_info=True)
                continue  # Skip failed PDFs, don't crash entire process

        # Check for deleted PDFs
        deleted_files = set(self.metadata.keys()) - set(pdf_files)
        if deleted_files:
            print(f"\nWarning: {len(deleted_files)} PDF(s) deleted since last run.")
            print("Consider rebuilding index from scratch by deleting metadata file.")
            for deleted in deleted_files:
                del self.metadata[deleted]

        if not files_to_process:
            print("No new or modified PDFs found. Index is up to date!")
            return

        print(f"\nProcessing {len(files_to_process)} PDF(s)...")

        # Create embeddings for new chunks
        if new_chunks:
            print(f"Creating embeddings for {len(new_chunks)} new chunks...")
            new_embeddings = self.model.encode(new_chunks, show_progress_bar=True)

            # Deduplicate against existing chunks
            print("Deduplicating chunks...")
            unique_chunks, unique_embeddings = self.deduplicate_chunks(
                new_chunks, new_embeddings, existing_embeddings
            )

            print(f"After deduplication: {len(unique_chunks)} unique chunks (removed {len(new_chunks) - len(unique_chunks)} duplicates)")

            # Combine with existing data
            all_chunks = existing_chunks + unique_chunks
            if existing_embeddings is not None:
                all_embeddings = np.vstack([existing_embeddings, unique_embeddings])
            else:
                all_embeddings = unique_embeddings

            # Build new index
            print("Building search index...")
            dimension = all_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(all_embeddings)
            index.add(all_embeddings.astype('float32'))

            # Save everything
            with open(CHUNKS_FILE, 'wb') as f:
                pickle.dump(all_chunks, f)
            faiss.write_index(index, INDEX_FILE)
            np.save(EMBEDDINGS_FILE, all_embeddings)  # Save embeddings as numpy array
            self.save_metadata()

            print(f"\n✓ Successfully processed! Total chunks in index: {len(all_chunks)}")

    def test_search(self, query, k=3):
        """Test search functionality"""
        if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
            print("Index not found. Please run processing first.")
            return

        # Load index and chunks
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, 'rb') as f:
            chunks = pickle.load(f)

        # Search
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb.astype('float32'), k=k)

        print(f"\nTop {k} matches for: '{query}'")
        print("-" * 80)
        for idx, (score, chunk_idx) in enumerate(zip(D[0], I[0])):
            print(f"\n{idx + 1}. Similarity: {score:.4f}")
            print(f"{chunks[chunk_idx][:300]}...")

# Main execution
if __name__ == "__main__":
    processor = IncrementalPDFProcessor()

    # Process PDFs (only new/modified ones)
    processor.process_pdfs()

    print("\n✓ Processor completed successfully!")
