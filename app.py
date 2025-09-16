import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

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

# Load API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier, fast

# Cache data loading to avoid reloading on every app run
@st.cache_resource
def load_processed_data():
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    index = faiss.read_index('index.faiss')
    return chunks, index

chunks, index = load_processed_data()

# Cache the embedding model to avoid reloading
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# Custom CSS with new Uiverse button styling
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

/* Full width containers */
.block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Input field focus color change - more specific targeting */
.stTextInput > div > div > input:focus {
    border-color: #7A5AF8 !important;
    box-shadow: 0 0 0 2px rgba(122, 90, 248, 0.2) !important;
}

/* Alternative targeting for input focus */
input[data-baseweb="input"]:focus {
    border-color: #7A5AF8 !important;
    box-shadow: 0 0 0 2px rgba(122, 90, 248, 0.2) !important;
}

/* Even more specific targeting */
div[data-testid="stTextInput"] input:focus {
    border-color: #7A5AF8 !important;
    box-shadow: 0 0 0 2px rgba(122, 90, 248, 0.2) !important;
}

/* Uiverse.io Button Styles */
.button {
  --h-button: 48px;
  --w-button: 102px;
  --round: 0.75rem;
  cursor: pointer;
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: all 0.25s ease;
  background: radial-gradient(
      65.28% 65.28% at 50% 100%,
      rgba(223, 113, 255, 0.8) 0%,
      rgba(223, 113, 255, 0) 100%
    ),
    linear-gradient(0deg, #7a5af8, #7a5af8);
  border-radius: var(--round);
  border: none;
  outline: none;
  padding: 12px 18px;
}
.button::before,
.button::after {
  content: "";
  position: absolute;
  inset: var(--space);
  transition: all 0.5s ease-in-out;
  border-radius: calc(var(--round) - var(--space));
  z-index: 0;
}
.button::before {
  --space: 1px;
  background: linear-gradient(
    177.95deg,
    rgba(255, 255, 255, 0.19) 0%,
    rgba(255, 255, 255, 0) 100%
  );
}
.button::after {
  --space: 2px;
  background: radial-gradient(
      65.28% 65.28% at 50% 100%,
      rgba(223, 113, 255, 0.8) 0%,
      rgba(223, 113, 255, 0) 100%
    ),
    linear-gradient(0deg, #7a5af8, #7a5af8);
}
.button:active {
  transform: scale(0.95);
}

.fold {
  z-index: 1;
  position: absolute;
  top: 0;
  right: 0;
  height: 1rem;
  width: 1rem;
  display: inline-block;
  transition: all 0.5s ease-in-out;
  background: radial-gradient(
    100% 75% at 55%,
    rgba(223, 113, 255, 0.8) 0%,
    rgba(223, 113, 255, 0) 100%
  );
  box-shadow: 0 0 3px black;
  border-bottom-left-radius: 0.5rem;
  border-top-right-radius: var(--round);
}
.fold::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  width: 150%;
  height: 150%;
  transform: rotate(45deg) translateX(0%) translateY(-18px);
  background-color: #e8e8e8;
  pointer-events: none;
}
.button:hover .fold {
  margin-top: -1rem;
  margin-right: -1rem;
}

.points_wrapper {
  overflow: hidden;
  width: 100%;
  height: 100%;
  pointer-events: none;
  position: absolute;
  z-index: 1;
}

.points_wrapper .point {
  bottom: -10px;
  position: absolute;
  animation: floating-points infinite ease-in-out;
  pointer-events: none;
  width: 2px;
  height: 2px;
  background-color: #fff;
  border-radius: 9999px;
}
@keyframes floating-points {
  0% {
    transform: translateY(0);
  }
  85% {
    opacity: 0;
  }
  100% {
    transform: translateY(-55px);
    opacity: 0;
  }
}
.points_wrapper .point:nth-child(1) {
  left: 10%;
  opacity: 1;
  animation-duration: 2.35s;
  animation-delay: 0.2s;
}
.points_wrapper .point:nth-child(2) {
  left: 30%;
  opacity: 0.7;
  animation-duration: 2.5s;
  animation-delay: 0.5s;
}
.points_wrapper .point:nth-child(3) {
  left: 25%;
  opacity: 0.8;
  animation-duration: 2.2s;
  animation-delay: 0.1s;
}
.points_wrapper .point:nth-child(4) {
  left: 44%;
  opacity: 0.6;
  animation-duration: 2.05s;
}
.points_wrapper .point:nth-child(5) {
  left: 50%;
  opacity: 1;
  animation-duration: 1.9s;
}
.points_wrapper .point:nth-child(6) {
  left: 75%;
  opacity: 0.5;
  animation-duration: 1.5s;
  animation-delay: 1.5s;
}
.points_wrapper .point:nth-child(7) {
  left: 88%;
  opacity: 0.9;
  animation-duration: 2.2s;
  animation-delay: 0.2s;
}
.points_wrapper .point:nth-child(8) {
  left: 58%;
  opacity: 0.8;
  animation-duration: 2.25s;
  animation-delay: 0.2s;
}
.points_wrapper .point:nth-child(9) {
  left: 98%;
  opacity: 0.6;
  animation-duration: 2.6s;
  animation-delay: 0.1s;
}
.points_wrapper .point:nth-child(10) {
  left: 65%;
  opacity: 1;
  animation-duration: 2.5s;
  animation-delay: 0.2s;
}

.inner {
  z-index: 2;
  gap: 6px;
  position: relative;
  width: 100%;
  color: white;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  font-weight: 500;
  line-height: 1.5;
  transition: color 0.2s ease-in-out;
}

.inner svg.icon {
  width: 18px;
  height: 18px;
  transition: fill 0.1s linear;
}

.button:focus svg.icon {
  fill: white;
}
.button:hover svg.icon {
  fill: transparent;
  animation:
    dasharray 1s linear forwards,
    filled 0.1s linear forwards 0.95s;
}
@keyframes dasharray {
  from {
    stroke-dasharray: 0 0 0 0;
  }
  to {
    stroke-dasharray: 68 68 0 0;
  }
}
@keyframes filled {
  to {
    fill: white;
  }
}

/* Headings and text */
h1, h2, h3 {
    color: #1f2937;
    font-weight: 600;
}
.stMarkdown p {
    color: #4b5563;
    font-size: 16px;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding: 20px;
}

/* Full width answer container */
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

/* Chat history cards */
.stMarkdown > div > div {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

/* Consistent spinner spacing */
div[data-testid="stSpinner"] {
    margin: 20px 0;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and branding
with st.sidebar:
    st.header("Axis Guidelines AI")
    st.markdown("Your tool for quick answers from company guidelines.")
    # Optional: Add a logo (upload to repo and uncomment)
    # st.image("logo.png", width=180)
    st.markdown("**How to use:**")
    st.markdown("- Type a question about our guidelines.")
    st.markdown("- Click 'Get Answer' to view results.")
    st.markdown("- Click the 'X' to clear the input.")
    st.markdown("**Example questions:**")
    st.markdown("- What are the content design principles?")
    st.markdown("- How should headings be written?")
    st.markdown("---")
    st.markdown("**Contact**: subzero@freecharge.com")

# Main content
st.title("📄 Axis Guidelines AI Assistant")
st.markdown("Ask about our company guidelines and get instant, accurate answers from our PDFs.")

# Query input
query = st.text_input(
    "Ask a question about your guidelines:",
    placeholder="e.g., What are the content design principles?",
    help="Type your question and press Enter or click Get Answer",
    key="query_input"
)

# Custom functional button with JavaScript
st.markdown("""
<script>
function submitQuery() {
    // Get the query input value
    const queryInput = window.parent.document.querySelector('[data-testid="stTextInput"] input');
    if (queryInput && queryInput.value.trim() !== '') {
        // Trigger form submission by pressing Enter on the input
        const event = new KeyboardEvent('keydown', {
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            which: 13,
            bubbles: true
        });
        queryInput.dispatchEvent(event);
    }
}
</script>

<button type="button" class="button" onclick="submitQuery()">
  <span class="fold"></span>
  <div class="points_wrapper">
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
    <i class="point"></i>
  </div>
  <span class="inner">
    <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5">
      <polyline points="13.18 1.37 13.18 9.64 21.45 9.64 10.82 22.63 10.82 14.36 2.55 14.36 13.18 1.37"></polyline>
    </svg>
    Get Answer
  </span>
</button>
""", unsafe_allow_html=True)

# Process the query when there's input and it's different from previous
if query and query != st.session_state.get('last_query', ''):
    with st.spinner("Searching guidelines..."):
        try:
            # Check cache for embedding, generate if not found
            if query not in st.session_state.embedding_cache:
                query_emb = embed_model.encode([query])
                st.session_state.embedding_cache[query] = query_emb[0]
            else:
                query_emb = np.array([st.session_state.embedding_cache[query]])

            faiss.normalize_L2(query_emb)
            D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
            context = "\n\n".join([chunks[i] for i in I[0]])

            # Ask Gemini (temperature=0 for consistent answers)
            prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nGive a clear, concise answer."
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))

            # Save to history and update last query
            st.session_state.history.append((query, response.text))
            st.session_state.last_query = query

            # Display results
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <h3 style="margin: 0; display: inline;">✅ Answer</h3>
                <span style="cursor: help; color: #6b7280;" title="This answer is generated from your PDFs">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Full width answer container
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown(response.text)
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("📚 Sources")
            for i, chunk_idx in enumerate(I[0], 1):
                with st.expander(f"Source {i}"):
                    st.markdown(chunks[chunk_idx])

            st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check that your API key is configured correctly and the data files exist.")

# Display chat history
if st.session_state.history:
    st.subheader("📝 Recent Questions")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-3:])):  # Show last 3, most recent first
        with st.expander(f"Q: {q[:80]}{'...' if len(q) > 80 else ''}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")

# Separate Help section with proper spacing
st.markdown("---")
st.subheader("❓ Need Help?")
with st.expander("Click here for help and tips"):
    st.markdown("""
    - **Updating PDFs**: Add/remove PDFs in `test_pdfs`, run `processor.py`, and push to GitHub.
    - **Contact**: Reach out to subzero@freecharge.com for issues or new PDFs.
    - **Tips**: Ask specific questions for best results (e.g., 'What is the tone for content design?').
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for the Axis team | Last updated: September 2025")
