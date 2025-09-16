import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'MAX_HISTORY_SIZE': 10,
    'MAX_CACHE_SIZE': 100,
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'GEMINI_MODEL': 'gemini-1.5-flash',
    'MAX_QUERY_LENGTH': 500,
    'TOP_K_RESULTS': 3,
    'TEMPERATURE': 0.0,
}

class GuidelinesAI:
    """Main application class for the Guidelines AI system."""
    
    def __init__(self):
        self.chunks: Optional[List[str]] = None
        self.index: Optional[faiss.Index] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self.genai_model = None
        
    def initialize(self):
        """Initialize all components with proper error handling."""
        try:
            self._setup_genai()
            self._load_data()
            self._setup_session_state()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            st.error("Failed to initialize the application. Please check the logs.")
            st.stop()
    
    def _setup_genai(self):
        """Configure Google Generative AI with validation."""
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in secrets")
            
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel(CONFIG['GEMINI_MODEL'])
            logger.info("Google Generative AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Generative AI: {e}")
            raise
    
    @st.cache_resource
    def _load_processed_data(_self):
        """Load processed chunks and FAISS index with error handling."""
        try:
            chunks_path = Path('chunks.pkl')
            index_path = Path('index.faiss')
            
            if not chunks_path.exists() or not index_path.exists():
                raise FileNotFoundError("Required data files not found")
            
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            index = faiss.read_index(str(index_path))
            
            logger.info(f"Loaded {len(chunks)} chunks and FAISS index")
            return chunks, index
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise
    
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load sentence transformer model with caching."""
        try:
            model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
            logger.info(f"Loaded embedding model: {CONFIG['EMBEDDING_MODEL']}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_data(self):
        """Load all required data."""
        self.chunks, self.index = self._load_processed_data()
        self.embed_model = self._load_embedding_model()
    
    def _setup_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'query': "",
            'history': [],
            'last_query': "",
            'embedding_cache': {},
            'last_query_time': 0,
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _validate_query(self, query: str) -> bool:
        """Validate user query."""
        if not query or not query.strip():
            return False
        if len(query) > CONFIG['MAX_QUERY_LENGTH']:
            st.error(f"Query too long. Maximum {CONFIG['MAX_QUERY_LENGTH']} characters allowed.")
            return False
        return True
    
    def _manage_cache_size(self):
        """Manage cache and history size to prevent memory issues."""
        # Limit embedding cache size
        if len(st.session_state.embedding_cache) > CONFIG['MAX_CACHE_SIZE']:
            # Remove oldest entries (simple FIFO)
            items = list(st.session_state.embedding_cache.items())
            st.session_state.embedding_cache = dict(items[-CONFIG['MAX_CACHE_SIZE']//2:])
        
        # Limit history size
        if len(st.session_state.history) > CONFIG['MAX_HISTORY_SIZE']:
            st.session_state.history = st.session_state.history[-CONFIG['MAX_HISTORY_SIZE']:]
    
    def _get_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query with caching."""
        if query not in st.session_state.embedding_cache:
            embedding = self.embed_model.encode([query])[0]
            st.session_state.embedding_cache[query] = embedding
        
        return np.array([st.session_state.embedding_cache[query]])
    
    def _search_similar_chunks(self, query_embedding: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Search for similar chunks using FAISS."""
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k=CONFIG['TOP_K_RESULTS']
        )
        
        context_chunks = [self.chunks[i] for i in indices[0]]
        return context_chunks, indices[0]
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini."""
        prompt = (
            f"Answer based ONLY on this context from our guidelines:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Give a clear, concise answer\n"
            f"- Only use information from the provided context\n"
            f"- If the context doesn't contain relevant information, say so\n"
            f"- Be specific and actionable when possible"
        )
        
        try:
            response = self.genai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=CONFIG['TEMPERATURE'],
                    max_output_tokens=1000,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def process_query(self, query: str) -> bool:
        """Process user query and return success status."""
        # Rate limiting (simple implementation)
        current_time = time.time()
        if current_time - st.session_state.last_query_time < 1.0:  # 1 second cooldown
            st.warning("Please wait a moment before submitting another query.")
            return False
        
        if not self._validate_query(query):
            return False
        
        # Skip if same as last query
        if query == st.session_state.last_query:
            return False
        
        try:
            with st.spinner("🔍 Searching guidelines..."):
                # Get embedding and search
                query_embedding = self._get_embedding(query)
                context_chunks, chunk_indices = self._search_similar_chunks(query_embedding)
                context = "\n\n".join(context_chunks)
                
                # Generate response
                response = self._generate_response(query, context)
                
                # Update session state
                st.session_state.history.append((query, response, chunk_indices))
                st.session_state.last_query = query
                st.session_state.last_query_time = current_time
                
                # Manage memory
                self._manage_cache_size()
                
                # Display results
                self._display_results(query, response, context_chunks, chunk_indices)
                
            return True
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            st.error("Sorry, something went wrong processing your query. Please try again.")
            return False
    
    def _display_results(self, query: str, response: str, context_chunks: List[str], chunk_indices: np.ndarray):
        """Display query results."""
        # Main answer
        st.markdown("### ✅ Answer")
        with st.container():
            st.markdown(f"**Question:** {query}")
            st.markdown("**Answer:**")
            st.markdown(response)
        
        # Sources
        st.markdown("### 📚 Sources")
        for i, (chunk, idx) in enumerate(zip(context_chunks, chunk_indices), 1):
            with st.expander(f"Source {i} (Chunk {idx})"):
                st.markdown(chunk)
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render application sidebar."""
        with st.sidebar:
            st.header("📄 Axis Guidelines AI")
            st.markdown("Your AI assistant for company guidelines and policies.")
            
            st.markdown("### 🚀 How to use")
            st.markdown("""
            1. Type your question in the search box
            2. Click 'Get Answer' or press Enter
            3. Review the answer and sources
            """)
            
            st.markdown("### 💡 Example questions")
            example_questions = [
                "What are the content design principles?",
                "How should headings be written?",
                "What is the brand voice and tone?",
                "What are the accessibility guidelines?"
            ]
            
            for question in example_questions:
                if st.button(f"💬 {question}", key=f"example_{hash(question)}"):
                    st.session_state.query_input = question
                    st.rerun()
            
            st.markdown("---")
            st.markdown("**📞 Support:** subzero@freecharge.com")
            
            # Cache statistics (for debugging)
            if st.checkbox("Show Debug Info"):
                st.markdown("### 🔧 Debug Info")
                st.write(f"Cache size: {len(st.session_state.embedding_cache)}")
                st.write(f"History size: {len(st.session_state.history)}")
                st.write(f"Chunks loaded: {len(self.chunks) if self.chunks else 0}")
    
    def render_history(self):
        """Render query history."""
        if not st.session_state.history:
            return
        
        st.markdown("### 📝 Recent Questions")
        
        # Show most recent queries first
        for i, (query, answer, _) in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q: {query[:60]}{'...' if len(query) > 60 else ''}"):
                st.markdown(f"**Question:** {query}")
                st.markdown(f"**Answer:** {answer}")
    
    def render_help(self):
        """Render help section."""
        st.markdown("### ❓ Help & Tips")
        with st.expander("Click here for help and tips"):
            st.markdown("""
            #### 🎯 **Getting Better Results**
            - Be specific in your questions
            - Use keywords from the guidelines
            - Ask about policies, procedures, or standards
            
            #### 🔄 **Updating Content**
            - Add/remove PDFs in the `test_pdfs` folder
            - Run `processor.py` to reprocess documents  
            - Deploy changes to update the app
            
            #### 📞 **Support**
            - Technical issues: subzero@freecharge.com
            - Missing content: Contact your team lead
            - Feature requests: Create an issue in the repository
            
            #### 🚨 **Troubleshooting**
            - If you get no results, try rephrasing your question
            - For slow responses, wait a moment and try again
            - Clear your browser cache if experiencing issues
            """)

def apply_custom_styles():
    """Apply custom CSS styles."""
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
        background: linear-gradient(135deg, #6d28d9 0%, #9333ea 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Container styling */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 2rem 1rem;
    }
    
    /* Success/error message styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Code and pre blocks */
    code {
        background-color: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        color: #7c3aed !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Axis Guidelines AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    apply_custom_styles()
    
    # Initialize application
    app = GuidelinesAI()
    app.initialize()
    
    # Render sidebar
    app.render_sidebar()
    
    # Main content
    st.title("📄 Axis Guidelines AI Assistant")
    st.markdown("""
    Ask questions about company guidelines and get instant, accurate answers powered by AI.
    Our system searches through your documents to provide relevant, contextual responses.
    """)
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about your guidelines:",
            placeholder="e.g., What are the content design principles?",
            help="Type your question and press Enter or click Get Answer",
            key="query_input",
            max_chars=CONFIG['MAX_QUERY_LENGTH']
        )
    
    with col2:
        st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if query:
                app.process_query(query)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process query on Enter key
    if query and query != st.session_state.get('last_displayed_query', ''):
        app.process_query(query)
        st.session_state.last_displayed_query = query
    
    # Display history
    app.render_history()
    
    # Display help
    app.render_help()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px 0;'>
        Built with ❤️ for the Axis team | Powered by Google Gemini & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()