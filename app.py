import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load API key and files
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier, fast

# Load your processed data
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('index.faiss')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🚀 Company Guidelines AI Assistant")
st.write("Ask questions about our policies—powered by your PDFs!")

# User input
query = st.text_input("Your question:", placeholder="e.g., What's the content design principles?")

if query:
    # Search for relevant chunks
    query_emb = embed_model.encode([query])
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    
    # Ask Gemini (temperature=0 for consistent answers)
    prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nGive a clear, concise answer."
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    
    st.subheader("Answer:")
    st.write(response.text)
    st.subheader("Sources:")
    for i in I[0]:
        st.write(f"• {chunks[i][:150]}...")

# Footer
st.write("---")
st.write("Built with ❤️ for the team | Update docs? Re-run processor.py")