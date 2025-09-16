import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Set page config for a modern, wide layout
st.set_page_config(page_title="Axis Guidelines AI", page_icon="📄", layout="wide")

# Load API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier, fast

# Load processed data
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('index.faiss')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS inspired by shadcn/ui with new button styling
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

/* Container for input and button */
.input-container {
    position: relative;
    width: 100%;
    max-width: 600px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Custom text input */
.clearable-input {
    width: 100%;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 40px 10px 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    font-size: 16px;
    color: #1f2937;
    outline: none;
}
.clearable-input:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

/* Clear button (X icon) */
.clear-button {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background-color: #e2e8f0;
    color: #1f2937;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    font-size: 14px;
    cursor: pointer;
    display: none; /* Hidden by default */
    align-items: center;
    justify-content: center;
}
.clearable-input:not(:placeholder-shown) ~ .clear-button {
    display: flex; /* Show when input has text */
}

/* Custom submit button (Uiverse.io style) */
.custom-button {
    font-family: Arial, Helvetica, sans-serif;
    font-weight: bold;
    color: white;
    background-color: #171717;
    padding: 1em 2em;
    border: none;
    border-radius: 0.6rem;
    position: relative;
    cursor: pointer;
    overflow: hidden;
}
.custom-button span:not(:nth-child(6)) {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    height: 30px;
    width: 30px;
    background-color: #97144D;
    border-radius: 50%;
    transition: 0.6s ease;
}
.custom-button span:nth-child(6) {
    position: relative;
}
.custom-button span:nth-child(1) {
    transform: translate(-3.3em, -4em);
}
.custom-button span:nth-child(2) {
    transform: translate(-6em, 1.3em);
}
.custom-button span:nth-child(3) {
    transform: translate(-0.2em, 1.8em);
}
.custom-button span:nth-child(4) {
    transform: translate(3.5em, 1.4em);
}
.custom-button span:nth-child(5) {
    transform: translate(3.5em, -3.8em);
}
.custom-button:hover span:not(:nth-child(6)) {
    transform: translate(-50%, -50%) scale(4);
    transition: 1.5s ease;
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

/* Chat history cards */
.stMarkdown > div > div {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
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
    st.markdown("**Contact**: policyteam@axis.com")

# Main content
st.title("📄 Axis Guidelines AI Assistant")
st.markdown("Ask about our company guidelines and get instant, accurate answers from our PDFs.")

# Initialize session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Custom clearable input with submit button
st.markdown("""
<div class="input-container">
    <input type="text" class="clearable-input" id="query-input" placeholder="e.g., What are the content design principles?" value="{0}">
    <button class="clear-button" onclick="document.getElementById('query-input').value='';Streamlit.setComponentValue('')">✕</button>
    <button class="custom-button" onclick="Streamlit.setComponentValue(document.getElementById('query-input').value);Streamlit.setTrigger('submit')">
        <span class="circle1"></span>
        <span class="circle2"></span>
        <span class="circle3"></span>
        <span class="circle4"></span>
        <span class="circle5"></span>
        <span class="text">Get Answer</span>
    </button>
</div>
<script>
document.getElementById('query-input').addEventListener('input', function(e) {
    Streamlit.setComponentValue(e.target.value);
});
</script>
""".format(st.session_state.query), unsafe_allow_html=True)

# Form to handle submit trigger
with st.form(key="question_form"):
    query = st.session_state.query
    if st.form_submit_button("Submit", on_click=None):  # Hidden trigger
        with st.spinner("Searching guidelines..."):
            # Search for relevant chunks
            query_emb = embed_model.encode([query])
            faiss.normalize_L2(query_emb)
            D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
            context = "\n\n".join([chunks[i] for i in I[0]])
            
            # Ask Gemini (temperature=0 for consistent answers)
            prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nGive a clear, concise answer."
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
            
            # Save to history
            st.session_state.history.append((query, response.text))
            
            # Clear the input
            st.session_state.query = ""
            
            # Display results
            st.subheader("Answer")
            st.markdown(response.text, help="This answer is generated from your PDFs.")
            
            st.subheader("Sources")
            for i in I[0]:
                st.markdown(f"• {chunks[i][:150]}...")
            
            st.markdown("---")

# Display chat history
if 'history' in st.session_state and st.session_state.history:
    st.subheader("Recent Questions")
    for q, a in st.session_state.history[-3:]:  # Show last 3
        st.markdown(f"**Question:** {q}\n\n**Answer:** {a}")

# Help section (collapsible)
with st.expander("Need Help?"):
    st.markdown("""
    - **Updating PDFs**: Add/remove PDFs in `test_pdfs`, run `processor.py`, and push to GitHub.
    - **Contact**: Reach out to policyteam@axis.com for issues or new PDFs.
    - **Tips**: Ask specific questions for best results (e.g., 'What is the tone for content design?').
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for the Axis team | Last updated: September 2025")