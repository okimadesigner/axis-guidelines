import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import logging
import subprocess
import sys
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config for a modern, wide layout
st.set_page_config(page_title="Axis Guidelines AI", page_icon="📄", layout="wide")

# Initialize session state variables FIRST
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'embedding_cache' not in st.session_state:
    st.session_state.embedding_cache = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Load API key from Streamlit secrets with validation
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        raise ValueError("API key is empty")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except KeyError:
    st.error("❌ GOOGLE_API_KEY not found in secrets. Please configure your API key.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {str(e)}")
    st.stop()

# Cache data loading to avoid reloading on every app run
@st.cache_resource
def load_processed_data():
    try:
        with open('chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
        index = faiss.read_index('index.faiss')
        logger.info(f"Successfully loaded {len(chunks)} chunks")
        return chunks, index
    except FileNotFoundError as e:
        st.error("❌ Required data files (chunks.pkl, index.faiss) not found. Please run the processor first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.stop()

chunks, index = load_processed_data()

# Cache the embedding model to avoid reloading
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {str(e)}")
        st.stop()

embed_model = load_embedding_model()

def manage_cache_size():
    """Prevent memory issues by limiting cache sizes."""
    if len(st.session_state.embedding_cache) > 100:
        items = list(st.session_state.embedding_cache.items())[-50:]
        st.session_state.embedding_cache = dict(items)

    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]

def validate_query(query: str) -> bool:
    """Validate user input."""
    if not query or not query.strip():
        return False
    if len(query) > 500:
        st.error("Query too long. Please keep it under 500 characters.")
        return False
    return True

def process_query(query: str) -> bool:
    """Process user query with proper error handling."""
    if not validate_query(query):
        return False

    if query == st.session_state.last_query:
        return False

    if st.session_state.processing:
        return False

    st.session_state.processing = True

    try:
        if query not in st.session_state.embedding_cache:
            query_emb = embed_model.encode([query])
            st.session_state.embedding_cache[query] = query_emb[0]
        else:
            query_emb = np.array([st.session_state.embedding_cache[query]])

        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb.astype('float32'), k=3)
        context = "\n\n".join([chunks[i] for i in I[0]])

        prompt = f"""Answer based ONLY on this context from our guidelines:

{context}

Question: {query}

Instructions:
- Give a clear, concise answer
- Only use information from the provided context
- If the context doesn't contain relevant information, say so clearly
- Be specific and actionable when possible"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0)
        )

        import re
        clean_response = re.sub(r'<[^>]+>', '', response.text)

        st.session_state.history.append((query, clean_response))
        st.session_state.last_query = query

        manage_cache_size()

        st.markdown(f"""
        <div class="answer-container">
            <h3 style="margin: 0 0 15px 0;">✅ Answer</h3>
            {clean_response}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📚 Sources")
        for i, chunk_idx in enumerate(I[0], 1):
            with st.expander(f"Source {i}"):
                st.markdown(chunks[chunk_idx])
        return True

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check that your API key is configured correctly and try again.")
        return False
    finally:
        st.session_state.processing = False

def reprocess_pdfs():
    """Run the processor script to update the index."""
    try:
        with st.spinner("🔄 Reprocessing PDFs... This may take a minute."):
            result = subprocess.run(
                [sys.executable, "processor.py"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                st.success("✅ PDFs reprocessed successfully!")
                st.info("Please refresh the page to load the updated index.")
                with st.expander("📋 Processing Log"):
                    st.code(result.stdout)
            else:
                st.error("❌ Reprocessing failed!")
                st.code(result.stderr)

    except subprocess.TimeoutExpired:
        st.error("⏱️ Processing timed out. Try processing fewer PDFs at once.")
    except Exception as e:
        st.error(f"❌ Error running processor: {str(e)}")

# Streamlined CSS - removed complex button animation
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Simple, clean button styling */
.stButton > button {
    background: linear-gradient(135deg, #7A5AF8 0%, #9B7FFF 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(122, 90, 248, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6A4AE8 0%, #8B6FEF 100%);
    box-shadow: 0 6px 12px rgba(122, 90, 248, 0.4);
    transform: translateY(-2px);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 4px rgba(122, 90, 248, 0.3);
}

/* Secondary button (for reprocess) */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
    box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
}

h1, h2, h3 {
    color: #1f2937;
    font-weight: 600;
}

.stMarkdown p {
    color: #4b5563;
    font-size: 16px;
}

.answer-container {
    width: 100% !important;
    max-width: 100% !important;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.stMarkdown > div > div {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

div[data-testid="stSpinner"] {
    margin: 20px 0;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Axis Guidelines AI")
    st.markdown("Your tool for quick answers from company guidelines.")

    st.markdown("**How to use:**")
    st.markdown("- Type a question about our guidelines")
    st.markdown("- Click 'Get Answer' to view results")
    st.markdown("- Use 'Reprocess PDFs' after adding new files")

    st.markdown("**Example questions:**")
    st.markdown("- What are the content design principles?")
    st.markdown("- How should headings be written?")

    st.markdown("---")

    # Reprocess button in sidebar
    st.markdown("**📂 Update Index**")
    if st.button("🔄 Reprocess PDFs", type="secondary", use_container_width=True):
        reprocess_pdfs()

    st.markdown("---")
    st.markdown("**Contact**: subzero@freecharge.com")

# Main content
st.title("📄 Axis Guidelines AI Assistant")
st.markdown("Ask about our company guidelines and get instant, accurate answers from our PDFs.")

# Query input with button in columns for better layout
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Ask a question about your guidelines:",
        placeholder="e.g., What are the content design principles?",
        help="Type your question and press Enter or click Get Answer",
        key="query_input",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
    if st.button("⚡ Get Answer", use_container_width=True):
        if query:
            with st.spinner("Searching guidelines..."):
                process_query(query)

# Auto-process on Enter key
if query and query != st.session_state.get('last_query', ''):
    with st.spinner("Searching guidelines..."):
        process_query(query)

# Display chat history
if st.session_state.history:
    st.subheader("📝 Recent Questions")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-3:])):
        with st.expander(f"Q: {q[:80]}{'...' if len(q) > 80 else ''}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")

# Help section
st.subheader("🌿 Need Help?")
with st.expander("Click here for help and tips"):
    st.markdown("""
    - **Adding PDFs**: Place PDFs in `test_pdfs` folder, then click 'Reprocess PDFs' in sidebar
    - **Updating PDFs**: Modify existing PDFs and reprocess to update the index
    - **Contact**: Reach out to subzero@freecharge.com for issues or new PDFs
    - **Tips**: Ask specific questions for best results (e.g., 'What is the tone for content design?')
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for the Axis team using Gemini 2.0 Flash | Last updated: October 2025")
