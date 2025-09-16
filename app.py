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

# Custom CSS for Swiggy-inspired styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #fff8f0 0%, #fef7ed 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #ffffff 0%, #fffbf7 100%);
    border-right: 1px solid #fed7aa;
    box-shadow: 0 0 20px rgba(151, 20, 77, 0.08);
}

.sidebar-header {
    background: linear-gradient(135deg, #97144D 0%, #be185d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
}

.sidebar-subtitle {
    color: #78716c;
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
    border: 1px solid #fed7aa;
    box-shadow: 0 1px 3px rgba(151, 20, 77, 0.08);
    transition: all 0.3s ease;
}

.sidebar-section:hover {
    box-shadow: 0 4px 12px rgba(151, 20, 77, 0.15);
    transform: translateY(-1px);
    border-color: #97144D;
}

.section-title {
    font-weight: 600;
    color: #292524;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
}

.section-content {
    color: #57534e;
    font-size: 0.85rem;
    line-height: 1.6;
}

.example-item {
    background: #fff8f0;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    border-left: 3px solid #97144D;
    font-size: 0.8rem;
    transition: all 0.2s ease;
    cursor: pointer;
}

.example-item:hover {
    background: #fed7aa;
    transform: translateX(4px);
    color: #97144D;
    font-weight: 500;
}

/* Main Content Styling */
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.main-header {
    text-align: left;
    margin-bottom: 3rem;
    padding: 2rem 0;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #97144D 0%, #be185d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    color: #78716c;
    font-size: 1.2rem;
    font-weight: 400;
}

/* Input Container */
.input-container {
    position: relative;
    margin-bottom: 2rem;
}

.input-wrapper {
    position: relative;
    display: flex;
    align-items: center;
}

/* Custom Input Styling */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 2px solid #fed7aa !important;
    border-radius: 16px !important;
    padding: 1.2rem 3.5rem 1.2rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(151, 20, 77, 0.08) !important;
    width: 100% !important;
    color: #292524 !important;
}

.stTextInput > div > div > input:focus {
    border-color: #97144D !important;
    box-shadow: 0 0 0 3px rgba(151, 20, 77, 0.12) !important;
    outline: none !important;
    background: #fffbf7 !important;
}

.stTextInput > div > div > input::placeholder {
    color: #a8a29e !important;
    font-weight: 400 !important;
}

/* Clear Icon */
.clear-btn {
    position: absolute;
    right: 1.2rem;
    top: 50%;
    transform: translateY(-50%);
    background: #f3f4f6;
    border: none;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 16px;
    color: #78716c;
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 10;
}

.clear-btn.show {
    opacity: 1;
}

.clear-btn:hover {
    background: #97144D;
    color: white;
    transform: translateY(-50%) scale(1.1);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #97144D 0%, #be185d 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.5rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(151, 20, 77, 0.25) !important;
    margin-top: 1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(151, 20, 77, 0.35) !important;
    background: linear-gradient(135deg, #be185d 0%, #97144D 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Response Cards */
.response-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid #fed7aa;
    box-shadow: 0 4px 16px rgba(151, 20, 77, 0.12);
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
    color: #292524;
    margin-bottom: 1.5rem;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 2px solid #fed7aa;
    padding-bottom: 0.8rem;
}

.response-content {
    color: #44403c;
    line-height: 1.8;
    font-size: 1.05rem;
}

.source-item {
    background: #fff8f0;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 1rem 0;
    border-left: 4px solid #97144D;
    color: #57534e;
    font-size: 0.95rem;
    line-height: 1.6;
    transition: all 0.3s ease;
}

.source-item:hover {
    background: #fed7aa;
    transform: translateX(6px);
    box-shadow: 0 2px 8px rgba(151, 20, 77, 0.1);
}

/* Help Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #97144D 0%, #be185d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: left;
    color: #a8a29e;
    font-size: 0.9rem;
    margin-top: 4rem;
    padding: 2rem 0;
    border-top: 1px solid #fed7aa;
}

/* Responsive */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.2rem;
    }
    
    .main-container {
        padding: 1rem;
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
document.addEventListener('DOMContentLoaded', function() {
    function addClearButton() {
        const inputs = document.querySelectorAll('input[type="text"]');
        inputs.forEach((input, index) => {
            if (!input.parentNode.querySelector('.clear-btn')) {
                const clearBtn = document.createElement('button');
                clearBtn.innerHTML = '×';
                clearBtn.className = 'clear-btn';
                clearBtn.type = 'button';
                
                clearBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    input.value = '';
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    clearBtn.classList.remove('show');
                    input.focus();
                });
                
                input.parentNode.style.position = 'relative';
                input.parentNode.appendChild(clearBtn);
                
                input.addEventListener('input', function() {
                    if (input.value.length > 0) {
                        clearBtn.classList.add('show');
                    } else {
                        clearBtn.classList.remove('show');
                    }
                });
                
                input.addEventListener('focus', function() {
                    if (input.value.length > 0) {
                        clearBtn.classList.add('show');
                    }
                });
            }
        });
    }
    
    // Add clear buttons periodically
    addClearButton();
    setInterval(addClearButton, 500);
});
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
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <div class="main-title">Guidelines AI Assistant</div>
    <div class="main-subtitle">Get instant, accurate answers from your company guidelines</div>
</div>
''', unsafe_allow_html=True)

# Input form
st.markdown('<div class="input-container">', unsafe_allow_html=True)

query = st.text_input(
    "", 
    placeholder="Ask me anything about our guidelines...", 
    key="question_input",
    label_visibility="collapsed"
)

submit_button = st.button("Get Answer", key="submit")

st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state for query tracking
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

if submit_button and query and query != st.session_state.last_query:
    st.session_state.last_query = query
    st.session_state.is_processing = True
    
    # Search for relevant chunks
    with st.spinner("Searching through guidelines..."):
        query_emb = embed_model.encode([query])
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
        context = "\n\n".join([chunks[i] for i in I[0]])
        
        # Ask Gemini
        prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nProvide a clear, comprehensive answer with actionable insights when applicable."
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    
    st.session_state.is_processing = False
    
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
        <div class="section-content">
    ''', unsafe_allow_html=True)
    
    for i in I[0]:
        st.markdown(f'<div class="source-item">{chunks[i][:200]}...</div>', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)