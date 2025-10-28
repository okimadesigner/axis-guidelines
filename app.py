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
from config import *

# Admin configuration - CHANGE THESE TO YOUR ADMIN EMAILS
ADMIN_EMAILS = [
    "subzero@freecharge.com",
    "your-second-admin@freecharge.com"  # Add more admins here
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Axis Guidelines AI", layout="centered")

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
    api_key = st.secrets.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found or empty in secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
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
        return SentenceTransformer(EMBEDDING_MODEL)
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

    # Validation
    if not validate_query(query):
        return False

    # Check if same query already processed
    if query == st.session_state.get('last_query', ''):
        st.info("This query was already processed. Check the answer above.")
        return False

    # Check if already processing
    if st.session_state.get('processing', False):
        st.warning("Already processing a query. Please wait...")
        return False

    # Set processing flag
    st.session_state.processing = True

    try:
        # Generate embedding
        if query not in st.session_state.embedding_cache:
            query_emb = embed_model.encode([query])
            st.session_state.embedding_cache[query] = query_emb[0]
        else:
            query_emb = np.array([st.session_state.embedding_cache[query]])

        # Search index
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb.astype('float32'), k=3)
        context = "\n\n".join([chunks[i] for i in I[0]])

        # Generate answer
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
            generation_config=genai.types.GenerationConfig(temperature=0),
            request_options={'timeout': 30}
        )

        # Clean response
        import re
        clean_response = re.sub(r'<[^>]+>', '', response.text)

        # Update session state
        st.session_state.history.append((query, clean_response))
        st.session_state.last_query = query
        manage_cache_size()

        # Success
        logger.info(f"Successfully processed query: {query[:50]}...")

        # Success
        st.session_state.processing = False
        return True

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check that your API key is configured correctly and try again.")
        st.session_state.processing = False
        return False

def system_health_check():
    """Check system health and data integrity"""
    checks = {
        'chunks_file': os.path.exists(CHUNKS_FILE),
        'index_file': os.path.exists(INDEX_FILE),
        'embeddings_file': os.path.exists(EMBEDDINGS_FILE),
        'pdf_folder': os.path.exists(PDF_FOLDER),
        'api_configured': bool(st.secrets.get("GOOGLE_API_KEY"))
    }

    if all(checks.values()):
        return True, "System healthy ✅"
    else:
        issues = [k for k, v in checks.items() if not v]
        return False, f"Issues found: {', '.join(issues)}"

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

# --- Custom CSS for clean, minimal UI ---
st.markdown("""
<style>
/* Hide textarea label */
.stTextArea > label { display: none !important; }

/* Textarea styling */
.stTextArea > div > div > textarea {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    padding: 12px !important;
}

/* Center container width */
.block-container {
    max-width: 850px;
    margin: 0 auto;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Answer box */
.answer-container {
    width: 100% !important;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0 0 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.answer-container p:last-child {
    margin-bottom: 0 !important;
}

/* Prevent button text wrapping */
.stButton button {
    white-space: nowrap !important;
    font-weight: 600 !important;
    background-color: #880e4f !important; /* Axis Maroon */
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
}
.stButton button:hover {
    background-color: #a51764 !important;
}

/* Professional Button Styling for secondary buttons */
.stButton > button[kind="secondary"] {
  background: #10B981 !important;
}

.stButton > button[kind="secondary"]:hover {
  background: #059669 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3) !important;
}

/* Logout link styling */
.logout-link {
  display: block;
  text-align: center;
  color: #97144D;
  font-weight: 500;
  margin-top: 8px;
  cursor: pointer;
}
.logout-link:hover {
  text-decoration: underline;
}

h1, h2, h3 {
    color: #1f2937;
    font-weight: 600;
}

.stMarkdown p {
    color: #4b5563;
    font-size: 16px;
}

div[data-testid="stSpinner"] {
    margin: 20px 0;
    text-align: center;
    width: 100%;
}

/* Divider line */
hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Custom Header ---
st.title("📄 Axis Guidelines AI")
st.markdown("*Your tool for quick answers from company guidelines.*")

# Sidebar
with st.sidebar:
    st.header("📄 Axis Guidelines AI")
    st.markdown("Quick answers from company guidelines.")

    st.markdown("**How to use:**")
    st.markdown("- Type a question about our guidelines")
    st.markdown("- Click 'Get Answer' to view results")

    st.markdown("**Example questions:**")
    st.markdown("- What are the content design principles?")
    st.markdown("- How should headings be written?")

    st.markdown("---")

    # Admin-only section for reprocessing
    st.markdown("**🔐 Admin Panel**")

    # Initialize admin session state
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        admin_email = st.text_input(
            "Admin Email",
            type="default",
            key="admin_email",
            placeholder="your-email@freecharge.com"
        )

        if st.button("🔓 Authenticate", use_container_width=True):
            if admin_email in ADMIN_EMAILS:
                st.session_state.admin_authenticated = True
                st.success("✅ Authenticated as admin!")
                st.rerun()
            else:
                st.error("❌ Unauthorized. Contact admin for access.")
    else:
        st.success(f"✅ Admin access granted")

        if st.button("🔄 Reprocess PDFs", type="secondary"):
            reprocess_pdfs()

        st.markdown('<p class="logout-link" onclick="window.location.reload()">← Logout</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**System Status**")
    healthy, status = system_health_check()
    if healthy:
        st.success(status)
    else:
        st.warning(status)

    st.markdown("---")
    st.markdown("**Contact**: subzero@freecharge.com")

# --- Input field ---
query = st.text_area("", placeholder="Ask your question here...")

# --- Left-aligned button ---
col1, _ = st.columns([1.5, 3.5])
with col1:
    submit = st.button("⚡ Get Answer", type="primary")

# --- Divider ---
st.markdown("<hr>", unsafe_allow_html=True)

# --- Answer container (Dynamic placeholder) ---
answer_placeholder = st.empty()

# --- When user submits ---
if submit:
    if not query or not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔍 Searching guidelines..."):
            success = process_query(query.strip())
            if success:
                # Get the latest answer from history
                if st.session_state.history:
                    query_text, answer_text = st.session_state.history[-1]
                    # 🪄 Update inside the container dynamically
                    answer_placeholder.markdown(f"""
                    <div class="answer-container">
                        <h4 style="margin-top:0;">🧠 Answer</h4>
                        <div style="white-space: pre-wrap; font-size: 16px;">{answer_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
