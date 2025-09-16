import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Set page config for a custom title and icon
st.set_page_config(page_title="Axis Guidelines AI", page_icon="✨", layout="wide")

# Load API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier, fast

# Load processed data
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('index.faiss')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS for modern minimal styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
    border-right: 1px solid #e2e8f0;
    box-shadow: 0 0 20px rgba(0,0,0,0.04);
}

.sidebar-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
}

.sidebar-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.5;
    margin-bottom: 2rem;
    text-align: center;
}

.sidebar-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.sidebar-section:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-1px);
}

.section-title {
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
}

.section-content {
    color: #475569;
    font-size: 0.85rem;
    line-height: 1.6;
}

.example-item {
    background: #f8fafc;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    border-left: 3px solid #667eea;
    font-size: 0.8rem;
    transition: all 0.2s ease;
}

.example-item:hover {
    background: #f1f5f9;
    transform: translateX(2px);
}

/* Main Content Styling */
.main-header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2rem 0;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    color: #64748b;
    font-size: 1.1rem;
    font-weight: 400;
}

/* Input Container */
.input-container {
    position: relative;
    max-width: 600px;
    margin: 0 auto 3rem auto;
}

/* Custom Input Styling */
.stTextInput > div > div > input {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 3rem 1rem 1.5rem;
    font-size: 1rem;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    outline: none;
}

/* Clear Icon */
.clear-icon {
    position: absolute;
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
    cursor: pointer;
    color: #94a3b8;
    font-size: 1.2rem;
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 10;
}

.clear-icon.show {
    opacity: 1;
}

.clear-icon:hover {
    color: #64748b;
    transform: translateY(-50%) scale(1.1);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    width: 100%;
    max-width: 200px;
    margin: 0 auto;
    display: block;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Response Cards */
.response-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    animation: slideIn 0.5s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.response-title {
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1rem;
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.response-content {
    color: #374151;
    line-height: 1.7;
    font-size: 1rem;
}

.source-item {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.75rem 0;
    border-left: 4px solid #667eea;
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.5;
    transition: all 0.3s ease;
}

.source-item:hover {
    background: #f1f5f9;
    transform: translateX(4px);
}

/* Help Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 4rem;
    padding: 2rem 0;
    border-top: 1px solid #e2e8f0;
}

/* Loading Animation */
.loading {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #e2e8f0;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .sidebar-section {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .response-card {
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
}
</style>

<script>
function addClearIcon() {
    const inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(input => {
        if (!input.parentNode.querySelector('.clear-icon')) {
            const clearIcon = document.createElement('div');
            clearIcon.innerHTML = '×';
            clearIcon.className = 'clear-icon';
            clearIcon.addEventListener('click', () => {
                input.value = '';
                input.dispatchEvent(new Event('input'));
                clearIcon.classList.remove('show');
            });
            
            input.parentNode.style.position = 'relative';
            input.parentNode.appendChild(clearIcon);
            
            input.addEventListener('input', () => {
                if (input.value.length > 0) {
                    clearIcon.classList.add('show');
                } else {
                    clearIcon.classList.remove('show');
                }
            });
        }
    });
}

// Run after page loads
setTimeout(addClearIcon, 100);
// Run periodically to catch dynamic content
setInterval(addClearIcon, 500);
</script>
""", unsafe_allow_html=True)

# Sidebar for navigation and branding
with st.sidebar:
    st.markdown('<div class="sidebar-header">✨ Axis Guidelines AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Intelligent answers from your company guidelines</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="sidebar-section">
        <div class="section-title">🚀 How to Use</div>
        <div class="section-content">
            Simply type your question and get instant answers from our comprehensive guidelines database.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="sidebar-section">
        <div class="section-title">💡 Example Questions</div>
        <div class="section-content">
            <div class="example-item">What are the content design principles?</div>
            <div class="example-item">How should headings be written?</div>
            <div class="example-item">What's our brand voice guidelines?</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="sidebar-section">
        <div class="section-title">🎯 Tips for Best Results</div>
        <div class="section-content">
            • Ask specific, focused questions<br>
            • Use keywords from our guidelines<br>
            • Check the sources for complete context
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Main content
st.markdown('''
<div class="main-header">
    <div class="main-title">Guidelines AI Assistant</div>
    <div class="main-subtitle">Get instant, accurate answers from your company guidelines</div>
</div>
''', unsafe_allow_html=True)

# Input form
query = st.text_input(
    "", 
    placeholder="Ask me anything about our guidelines...", 
    key="question_input",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("Get Answer", key="submit", use_container_width=True)

# Initialize session state for query tracking
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if submit_button and query and query != st.session_state.last_query:
    st.session_state.last_query = query
    
    # Show loading
    with st.spinner(""):
        st.markdown('<div class="loading"><div class="loading-spinner"></div></div>', unsafe_allow_html=True)
    
    # Search for relevant chunks
    query_emb = embed_model.encode([query])
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    
    # Ask Gemini
    prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nProvide a clear, comprehensive answer with actionable insights when applicable."
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    
    # Display results
    st.markdown('''
    <div class="response-card">
        <div class="response-title">💡 Answer</div>
        <div class="response-content">{}</div>
    </div>
    '''.format(response.text), unsafe_allow_html=True)
    
    st.markdown('''
    <div class="response-card">
        <div class="response-title">📚 Sources</div>
    ''', unsafe_allow_html=True)
    
    for i in I[0]:
        st.markdown(f'<div class="source-item">{chunks[i][:200]}...</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Help section
with st.expander("🔧 Need Help?"):
    st.markdown("""
    **Updating Guidelines**: Add new PDFs to `test_pdfs`, run `processor.py`, and deploy updates.
    
    **Best Practices**: 
    - Ask specific questions for more accurate results
    - Use terminology from your company guidelines
    - Review sources for complete context
    
    **Support**: Contact your admin team for technical issues or new guideline additions.
    """)

# Footer
st.markdown('''
<div class="footer">
    Built with ❤️ for the Axis team • Powered by AI • Last updated September 2025
</div>
''', unsafe_allow_html=True)