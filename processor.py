import os
import hashlib
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import json
from datetime import datetime

# Configuration
PDF_FOLDER = 'test_pdfs'
METADATA_FILE = 'pdf_metadata.json'
CHUNKS_FILE = 'chunks.pkl'
INDEX_FILE = 'index.faiss'
EMBEDDINGS_FILE = 'embeddings.npy'  # New file for embeddings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.95  # For deduplication (95% similar = duplicate)

class IncrementalPDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

    def extract_chunks_from_pdf(self, pdf_path):
        """Extract text chunks from a single PDF"""
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

        # Clean the extracted text to remove HTML-like artifacts and formatting
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common PDF artifacts
        text = re.sub(r'[•●○◆◇■□▪▫]', '', text)  # Remove bullet point symbols
        text = re.sub(r'[^\x20-\x7E\n]', '', text)  # Keep only ASCII printable characters + newline

        # Split into overlapping chunks for better context
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if len(chunk) > 50:  # Skip very short chunks
                chunks.append(chunk)
        return chunks

    def compute_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def deduplicate_chunks(self, new_chunks, new_embeddings, existing_embeddings=None):
        """Remove duplicate chunks based on semantic similarity"""
        if len(new_chunks) == 0:
            return [], np.array([])

        unique_chunks = []
        unique_embeddings = []

        # Build combined embedding set for comparison
        all_embeddings = list(existing_embeddings) if existing_embeddings is not None else []

        for i, (chunk, embedding) in enumerate(zip(new_chunks, new_embeddings)):
            is_duplicate = False

            # Check against all previously added embeddings
            for existing_emb in all_embeddings:
                similarity = self.compute_similarity(embedding, existing_emb)
                if similarity > SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)
                unique_embeddings.append(embedding)
                all_embeddings.append(embedding)

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

        # Identify which PDFs need processing
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            file_hash = self.get_file_hash(pdf_path)
            file_mtime = os.path.getmtime(pdf_path)

            # Check if file is new or modified
            if pdf_file not in self.metadata or self.metadata[pdf_file]['hash'] != file_hash:
                files_to_process.append(pdf_file)
                print(f"  - {pdf_file} (new or modified)")

                # Extract chunks from this PDF
                chunks = self.extract_chunks_from_pdf(pdf_path)
                new_chunks.extend(chunks)

                # Update metadata
                self.metadata[pdf_file] = {
                    'hash': file_hash,
                    'modified': file_mtime,
                    'processed': datetime.now().isoformat(),
                    'chunk_count': len(chunks)
                }

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

    # Quick test search
    print("\n" + "=" * 80)
    processor.test_search("What are the content design principles?", k=3)
