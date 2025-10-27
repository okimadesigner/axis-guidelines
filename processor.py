import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load your API key (we'll use it later for testing)
with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()

# Step 1: Read PDFs and extract text chunks
def process_pdfs(pdf_folder):
    all_chunks = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            reader = PdfReader(os.path.join(pdf_folder, pdf_file))
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            # Split into chunks (about 500 chars each for good results)
            chunks = [text[i:i+500] for i in range(0, len(text), 400)]
            all_chunks.extend(chunks)
    return all_chunks

# Step 2: Create embeddings (vectors)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, fast model (384 dims)
chunks = process_pdfs('test_pdfs')
embeddings = model.encode(chunks)

# Step 3: Build search index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
faiss.normalize_L2(embeddings)  # Normalize for better search
index.add(embeddings.astype('float32'))

# Save everything
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)
faiss.write_index(index, 'index.faiss')
print(f"Processed {len(chunks)} chunks from your PDFs! Saved to files.")

# Quick test: Search for a sample query
query = "What are the content design principles?"  # Change this to test
query_emb = model.encode([query])
faiss.normalize_L2(query_emb)
D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 matches
print("Top matches:")
for i in I[0]:
    print(chunks[i][:200] + "...")  # Show snippet